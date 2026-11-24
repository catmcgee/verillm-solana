//! Host tests of the pure logic. Off chain the Solana crates fall back to ordinary
//! implementations of the hash and curve syscalls, so the same code runs here.

use serde_json::Value;
use verillm::{math, state::Params, vrf};

fn load(file: &str) -> Value {
    serde_json::from_str(&std::fs::read_to_string(format!("{}/tests/{file}", env!("CARGO_MANIFEST_DIR"))).unwrap()).unwrap()
}

fn bytes(value: &Value) -> Vec<u8> {
    hex::decode(value.as_str().unwrap()).unwrap()
}

fn numbers(value: &Value) -> Vec<u32> {
    value.as_array().unwrap().iter().map(|number| number.as_u64().unwrap() as u32).collect()
}

#[test]
fn vrf_matches_rfc9381_vectors_and_rejects_tampering() {
    let data = load("../../../tests/rfc9381_edwards25519_tai.json");
    let vectors = data["vectors"].as_array().unwrap();
    assert_eq!(vectors.len(), 3);
    for vector in vectors {
        let key: [u8; 32] = bytes(&vector["pk"]).try_into().unwrap();
        let alpha = bytes(&vector["alpha"]);
        let proof: [u8; 80] = bytes(&vector["pi"]).try_into().unwrap();
        assert_eq!(vrf::verify(&key, &alpha, &proof).unwrap().to_vec(), bytes(&vector["beta"]));
        let mut other = alpha.clone();
        other.push(0);
        assert!(vrf::verify(&key, &other, &proof).is_none());
        for at in [0, 5, 31, 32, 40, 47, 48, 70, 79] {
            let mut bad = proof;
            bad[at] ^= 1;
            assert!(vrf::verify(&key, &alpha, &bad).is_none(), "tampered byte {at} accepted");
        }
        let mut wrong_key = key;
        wrong_key[3] ^= 4;
        assert!(vrf::verify(&wrong_key, &alpha, &proof).is_none());
    }
}

#[test]
fn stream_permutation_sampling_and_merkle_match_python() {
    let data = load("primitives.json");
    let seed = bytes(&data["seed"]);
    for case in data["permutations"].as_array().unwrap() {
        assert_eq!(math::permutation(&seed, case["size"].as_u64().unwrap() as u32), numbers(&case["order"]));
    }
    for case in data["samples"].as_array().unwrap() {
        let picks = math::sample(&seed, case["count"].as_u64().unwrap() as u32, case["width"].as_u64().unwrap() as u32);
        assert_eq!(picks, numbers(&case["indices"]));
    }
    let root: [u8; 32] = bytes(&data["merkle"]["root"]).try_into().unwrap();
    for opening in data["merkle"]["openings"].as_array().unwrap() {
        let index = opening["index"].as_u64().unwrap();
        let path: Vec<[u8; 32]> = opening["path"].as_array().unwrap().iter().map(|hash| bytes(hash).try_into().unwrap()).collect();
        let leaf = math::keccak256(&[&[0], &bytes(&opening["value"])]);
        assert!(math::merkle(&root, leaf, index, &path));
        assert!(!math::merkle(&root, leaf, index ^ 1, &path));
        assert!(!math::merkle(&root, math::keccak256(&[&[0], &[1, 2, 3, 4]]), index, &path));
        assert!(!math::merkle(&root, leaf, index, &path[..path.len() - 1]));
    }
}

fn params_with(epsilon: u128) -> Params {
    Params {
        verifiers: 3,
        reverifiers: 11,
        samples: 64,
        width: 896,
        theta_bps: 9500,
        pe_max_bps: 800,
        pm_min_bps: 7000,
        pw_min_bps: 7500,
        pm_steps: 7,
        pw_steps: 83886,
        reward: 6,
        slash: 2,
        dispute_cost: 8,
        phase_seconds: 3600,
        dispute_seconds: 3,
        unbonding_seconds: 604800,
        epsilon,
        mean_abs_max: (0.02 * 18446744073709551616.0) as u128,
        unit: 10_000_000,
        node_stake: 200_000_000,
        scheduler_stake: 1_000_000_000,
    }
}

// Sampled values from runs of the demo, with results cross-checked against an independent implementation.
#[test]
fn consistent_matches_recorded_results() {
    let data = load("consistent.json");
    let cases = data["cases"].as_array().unwrap();
    assert!(cases.len() > 20);
    for case in cases {
        let params = params_with(case["epsilon"].as_str().unwrap().parse().unwrap());
        let (ok, agreeing) = math::consistent(&params, &bytes(&case["a"]), &bytes(&case["b"]));
        assert_eq!((ok, agreeing as u64), (case["consistent"].as_bool().unwrap(), case["agreeing"].as_u64().unwrap()), "{}", case["source"]);
    }
}

fn floats(values: &[f32]) -> Vec<u8> {
    let mut all: Vec<f32> = values.to_vec();
    all.resize(64, 1.0);
    all.iter().flat_map(|value| value.to_bits().to_be_bytes()).collect()
}

#[test]
fn consistent_handles_the_edges_of_the_fixed_point_range() {
    let params = params_with(1 << 54);
    let same = floats(&[0.5, -3.25, 1e-30, 0.0, -0.0]);
    assert_eq!(math::consistent(&params, &same, &same), (true, 64));
    // Zero and negative zero are the same number and share an exponent field.
    assert_eq!(math::consistent(&params, &floats(&[0.0]), &floats(&[-0.0])), (true, 64));
    // Infinity, NaN and values above 2^88 saturate and never agree with a finite value.
    for huge in [f32::INFINITY, f32::NAN, 1e30, -1e30] {
        let (ok, agreeing) = math::consistent(&params, &floats(&[1.0]), &floats(&[huge]));
        assert_eq!((ok, agreeing), (false, 63), "{huge}");
    }
    // The largest value that still fits is compared exactly: the two are 2^64 apart.
    let big = f32::from_bits((213 << 23) | 1);
    let next = f32::from_bits((213 << 23) | 2);
    assert_eq!(math::consistent(&params, &floats(&[big]), &floats(&[next])).1, 63);
    assert_eq!(math::consistent(&params, &floats(&[big]), &floats(&[big])), (true, 64));
    // A difference just inside and just outside epsilon (2^-10 here).
    assert_eq!(math::consistent(&params, &floats(&[1.0]), &floats(&[1.0 + 0.0009])).1, 64);
    assert_eq!(math::consistent(&params, &floats(&[1.0]), &floats(&[1.0 + 0.0011])).1, 63);
}
