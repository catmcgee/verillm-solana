//! The verification program of VeriLLM (arXiv 2509.24257v4, Sections 5.2 to 5.7 and Appendix E).
//!
//! Computation and Merkle proof generation stay off chain. The chain holds compact digests,
//! VRF proofs and sampled Merkle openings, checks them, counts verdicts and moves stake.
//! Section references in the comments point at the paper.
//!
//! Three things follow from how Solana works:
//!
//!   - The sampled openings of one step are larger than a transaction, so they arrive in
//!     several. Each chunk is checked on arrival against the stored root and the indices this
//!     program derived; a last instruction closes the step once every sample is in.
//!   - Ed25519 signatures are checked by Solana's native Ed25519 program. The transaction
//!     carries that instruction and this program checks, through the instructions sysvar,
//!     that it covers exactly the expected key and message.
//!   - The VRF is ECVRF-EDWARDS25519-SHA512-TAI, verified here with the curve25519 syscalls.

#![allow(deprecated)] // the system program helpers have moved crates; the old paths still work

mod accounts;
mod adjudication;
pub mod math;
mod registry;
mod roles;
mod settlement;
pub mod state;
mod verification;
pub mod vrf;

use accounts::{event, fail};
use borsh::{BorshDeserialize, BorshSerialize};
use solana_program::{account_info::AccountInfo, entrypoint::ProgramResult, log::sol_log_data, program_error::ProgramError, pubkey::Pubkey};
use state::Params;

#[cfg(not(feature = "no-entrypoint"))]
solana_program::entrypoint!(process);

#[derive(BorshSerialize, BorshDeserialize)]
pub enum Instruction {
    Initialize { run: [u8; 8], params: Params, pool: u64 },
    RegisterScheduler { vrf_key: [u8; 32], signing_key: [u8; 32], endpoint: String },
    RegisterNode { model_id: [u8; 32], parameter_hash: [u8; 32], slice: u32, signing_key: [u8; 32], endpoint: String },
    Deregister,
    Withdraw,
    CreateTask { h: [u8; 32], model_id: [u8; 32], stages: u32 },
    AssignRoles { stage: u32, proof: [u8; 80] },
    CommitInference { steps: u32, output_hash: [u8; 32], trace_roots: Vec<[u8; 32]> },
    CommitVerification { root: [u8; 32] },
    CommitVerdict { commitment: [u8; 32] },
    BeginSampling { proof: [u8; 80], inferencer_root: [u8; 32], trace_path: Vec<[u8; 32]> },
    SamplingOpenings { start: u32, values: Vec<u8>, paths: Vec<[u8; 32]> },
    FinishSampling,
    BeginReveal { verdict: bool, randomness: [u8; 32] },
    RevealOpenings { start: u32, values: Vec<u8>, paths: Vec<[u8; 32]> },
    FinishReveal,
    Adjudicate,
    Dispute,
    Settle,
    // Checks one VRF proof and logs its output. Touches no account; used by the tests.
    VrfCheck { key: [u8; 32], alpha: Vec<u8>, proof: [u8; 80] },
}

pub fn process<'a>(program: &Pubkey, accounts: &'a [AccountInfo<'a>], data: &[u8]) -> ProgramResult {
    let instruction = Instruction::try_from_slice(data).map_err(|_| ProgramError::InvalidInstructionData)?;
    let accounts = &mut accounts.iter();
    match instruction {
        Instruction::Initialize { run, params, pool } => registry::initialize(program, accounts, run, params, pool),
        Instruction::RegisterScheduler { vrf_key, signing_key, endpoint } => {
            registry::register_scheduler(program, accounts, vrf_key, signing_key, endpoint)
        }
        Instruction::RegisterNode { model_id, parameter_hash, slice, signing_key, endpoint } => {
            registry::register_node(program, accounts, model_id, parameter_hash, slice, signing_key, endpoint)
        }
        Instruction::Deregister => registry::deregister(program, accounts),
        Instruction::Withdraw => registry::withdraw(program, accounts),
        Instruction::CreateTask { h, model_id, stages } => roles::create_task(program, accounts, h, model_id, stages),
        Instruction::AssignRoles { stage, proof } => roles::assign_roles(program, accounts, stage, &proof),
        Instruction::CommitInference { steps, output_hash, trace_roots } => {
            roles::commit_inference(program, accounts, steps, output_hash, &trace_roots)
        }
        Instruction::CommitVerification { root } => verification::commit_verification(program, accounts, root),
        Instruction::CommitVerdict { commitment } => verification::commit_verdict(program, accounts, commitment),
        Instruction::BeginSampling { proof, inferencer_root, trace_path } => {
            verification::begin_sampling(program, accounts, &proof, inferencer_root, &trace_path)
        }
        Instruction::SamplingOpenings { start, values, paths } => verification::sampling_openings(program, accounts, start, &values, &paths),
        Instruction::FinishSampling => verification::finish_sampling(program, accounts),
        Instruction::BeginReveal { verdict, randomness } => verification::begin_reveal(program, accounts, verdict, &randomness),
        Instruction::RevealOpenings { start, values, paths } => verification::reveal_openings(program, accounts, start, &values, &paths),
        Instruction::FinishReveal => verification::finish_reveal(program, accounts),
        Instruction::Adjudicate => adjudication::adjudicate(program, accounts),
        Instruction::Dispute => adjudication::dispute(program, accounts),
        Instruction::Settle => settlement::settle(program, accounts),
        Instruction::VrfCheck { key, alpha, proof } => {
            let beta = vrf::verify(&key, &alpha, &proof).ok_or_else(|| fail("VRF proof"))?;
            sol_log_data(&[&[event::VRF_CHECKED], &beta]);
            Ok(())
        }
    }
}
