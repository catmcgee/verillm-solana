//! Account layouts. Every account starts with its kind and the id of the run it belongs to.
//!
//! All accounts are program-derived addresses seeded with the run id, so each run of the demo
//! has its own registry, tasks and reward pool:
//!
//!   config  ["config", run]                         parameters, scheduler, pool; holds the lamports
//!   node    ["node", run, owner]                    one registered node and its stake
//!   group   ["group", run, model, slice]            members of one slice, in registration order
//!   task    ["task", run, task id]
//!   stage   ["stage", run, task id, stage]
//!   round   ["round", run, task id, stage, round]   committee, sampling and every vote

use borsh::{BorshDeserialize, BorshSerialize};
use solana_program::pubkey::Pubkey;

pub const CONFIG: u8 = 1;
pub const NODE: u8 = 2;
pub const GROUP: u8 = 3;
pub const TASK: u8 = 4;
pub const STAGE: u8 = 5;
pub const ROUND: u8 = 6;

pub const NONE: u8 = 0;
pub const ACCEPTED: u8 = 1;
pub const REJECTED: u8 = 2;
pub const INDETERMINATE: u8 = 3;

#[derive(BorshSerialize, BorshDeserialize, Clone)]
pub struct Params {
    pub verifiers: u32,   // m, odd (Section 5.7)
    pub reverifiers: u32, // m' > m, odd (Lemma E.2)
    pub samples: u32,     // n (Eq. 23)
    pub width: u32,       // scalars in one final-token hidden state
    pub theta_bps: u32,   // agreement rate theta of Eq. 29, in basis points
    pub pe_max_bps: u32,  // on-chain bit-aware thresholds (Section 7), in basis points
    pub pm_min_bps: u32,
    pub pw_min_bps: u32,
    pub pm_steps: u32,          // e_m, in mantissa steps
    pub pw_steps: u32,          // e_w, in mantissa steps
    pub reward: u32,            // 6 units (Appendix E)
    pub slash: u32,             // 2 units
    pub dispute_cost: u32,      // 8 units
    pub phase_seconds: i64,     // commit and reveal deadlines
    pub dispute_seconds: i64,   // window in which a rejected inferencer may dispute
    pub unbonding_seconds: i64, // delta_wait (Section 5.2)
    pub epsilon: u128,          // tolerance on |v - V| (Eq. 29), in units of 2^-64
    pub mean_abs_max: u128,     // bound on the mean discrepancy, in units of 2^-64
    pub unit: u64,              // lamports per reward unit
    pub node_stake: u64,        // collateral per node, lamports
    pub scheduler_stake: u64,   // collateral of the scheduler, lamports
}

#[derive(BorshSerialize, BorshDeserialize)]
pub struct Config {
    pub kind: u8,
    pub run: [u8; 8],
    pub params: Params,
    pub scheduler: Pubkey, // all zero until a scheduler registers
    pub vrf_key: [u8; 32],
    pub scheduler_signing_key: [u8; 32],
    pub scheduler_stake: u64,
    pub pool: u64,
    pub tasks: u64,
    pub scheduler_endpoint: String,
}

#[derive(BorshSerialize, BorshDeserialize)]
pub struct Node {
    pub kind: u8,
    pub run: [u8; 8],
    pub owner: Pubkey,
    pub model_id: [u8; 32],
    pub parameter_hash: [u8; 32],
    pub signing_key: [u8; 32], // Ed25519 public key
    pub slice: u32,
    pub exit_at: i64,
    pub active: bool,
    pub stake: u64,
    pub endpoint: String,
}

#[derive(BorshSerialize, BorshDeserialize)]
pub struct Group {
    pub kind: u8,
    pub run: [u8; 8],
    pub model_id: [u8; 32],
    pub slice: u32,
    pub members: Vec<Pubkey>,
}

#[derive(BorshSerialize, BorshDeserialize)]
pub struct Task {
    pub kind: u8,
    pub run: [u8; 8],
    pub id: u64,
    pub h: [u8; 32], // H(req)
    pub model_id: [u8; 32],
    pub output_hash: [u8; 32],
    pub stages: u32,
    pub steps: u32, // T
    pub inference_committed: bool,
}

#[derive(BorshSerialize, BorshDeserialize)]
pub struct Stage {
    pub kind: u8,
    pub run: [u8; 8],
    pub task: u64,
    pub stage: u32,
    pub inferencer: Pubkey,
    pub trace_root: [u8; 32], // commitment to every per-step root of this segment
    pub round: u8,            // 1 or 2
    pub reverify_requested: bool,
    pub dispute_paid: bool,
    pub settled: bool,
    pub dispute_deadline: i64,
}

#[derive(BorshSerialize, BorshDeserialize, Default)]
pub struct Vote {
    pub root: [u8; 32],
    pub commitment: [u8; 32],
    pub values: Vec<u8>, // the verifier's sampled scalars, 4 bytes each
    pub opened: Vec<u8>, // one flag per sample: its opening has arrived
    pub opened_count: u32,
    pub root_in: bool,
    pub verdict_in: bool,
    pub opening: bool, // the commitment is open and sample openings are arriving
    pub failed: bool,  // an opening did not hold
    pub revealed: bool,
    pub valid: bool,   // every Merkle opening held
    pub verdict: bool, // b
    pub agrees: bool,  // in the proper cluster and voted with it
    pub agreeing: u32, // samples within epsilon of the inferencer's
    pub consistent: bool,
}

#[derive(BorshSerialize, BorshDeserialize)]
pub struct Round {
    pub kind: u8,
    pub run: [u8; 8],
    pub task: u64,
    pub stage: u32,
    pub round: u8,
    pub committee: Vec<Pubkey>,
    pub role_beta: [u8; 64],
    pub sample_beta: [u8; 64],
    pub inferencer_root: [u8; 32],
    pub indices: Vec<u32>,         // the sampled positions, derived from sample_beta
    pub reference_values: Vec<u8>, // the inferencer's sampled scalars, 4 bytes each
    pub reference_opened: Vec<u8>,
    pub reference_count: u32,
    pub opened_at: i64,
    pub sampled_at: i64,
    pub roots_in: u32,
    pub verdicts_in: u32,
    pub reveals: u32,
    pub sampling: bool, // the VRF output is fixed and reference openings are arriving
    pub sampled: bool,  // every reference opening has been checked
    pub adjudicated: bool,
    pub cluster_verdict: bool,
    pub outcome: u8,
    pub cluster_size: u32,
    pub votes: Vec<Vote>, // one per committee member, in committee order
}
