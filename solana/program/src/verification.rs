//! The verifiers' commitments, the sampling of indices and the reveal (Sections 5.6 and 5.7).

use crate::accounts::{event, fail, load, member, now, require, scheduler_only, signer, store, Accounts};
use crate::math::{consistent, keccak256, merkle, sample};
use crate::state::{Config, Node, Round, Stage, Task, CONFIG, NODE, ROUND, STAGE, TASK};
use crate::vrf;
use solana_program::{
    account_info::{next_account_info, AccountInfo},
    ed25519_program,
    entrypoint::ProgramResult,
    log::sol_log_data,
    program_error::ProgramError,
    pubkey::Pubkey,
    sysvar::instructions as tx_instructions,
};

/// Checks that the instruction just before this one is a call to the native Ed25519 program
/// that verifies one signature by `key` over exactly `message`. The native program has
/// already checked the signature itself, or the transaction would not be running.
fn signed_by(sysvar: &AccountInfo, key: &[u8; 32], message: &[u8]) -> ProgramResult {
    let current = tx_instructions::load_current_index_checked(sysvar)? as usize;
    require!(current > 0, "signature");
    let instruction = tx_instructions::load_instruction_at_checked(current - 1, sysvar)?;
    require!(instruction.program_id == ed25519_program::ID, "signature");
    // One signature, with key, signature and message all inside this instruction's data:
    // a 16-byte header, the key at 16, the signature at 48 and the message at 112.
    let size = (message.len() as u16).to_le_bytes();
    let header = [1, 0, 48, 0, 0xff, 0xff, 16, 0, 0xff, 0xff, 112, 0, size[0], size[1], 0xff, 0xff];
    let data = &instruction.data;
    require!(data.len() == 112 + message.len() && data[..16] == header && data[16..48] == key[..] && data[112..] == *message, "signature");
    Ok(())
}

/// What a node signs: the task, the stage, the last step and a root (Section 5.4).
fn bound_message(task: &Task, stage: u32, root: &[u8; 32]) -> [u8; 72] {
    let mut message = [0u8; 72];
    message[..32].copy_from_slice(&task.h);
    message[32..36].copy_from_slice(&stage.to_be_bytes());
    message[36..40].copy_from_slice(&task.steps.to_be_bytes());
    message[40..].copy_from_slice(root);
    message
}

/// MerkleVerify for the sampled scalars at positions `start..` of the sample. `values` holds
/// 4 big-endian bytes per sample and `paths` the sibling hashes of each opening, one after
/// another. Returns one flag per opening.
fn check_openings(root: &[u8; 32], indices: &[u32], start: u32, values: &[u8], paths: &[[u8; 32]]) -> Result<Vec<bool>, ProgramError> {
    let count = values.len() / 4;
    require!(values.len().is_multiple_of(4) && count > 0 && paths.len().is_multiple_of(count), "openings malformed");
    require!(start as usize + count <= indices.len(), "openings out of range");
    let depth = paths.len() / count;
    Ok((0..count)
        .map(|j| {
            let leaf = keccak256(&[&[0], &values[4 * j..4 * j + 4]]);
            merkle(root, leaf, indices[start as usize + j] as u64, &paths[j * depth..(j + 1) * depth])
        })
        .collect())
}

/// Eqs. 19 and 20: the root over the final token's hidden state, signed with the registered key.
/// The instruction before this one must be the native Ed25519 check of that signature.
///
/// Accounts: verifier (signer), config, task, round, node record, instructions sysvar.
pub fn commit_verification(program: &Pubkey, accounts: Accounts, root: [u8; 32]) -> ProgramResult {
    let sender = signer(next_account_info(accounts)?)?;
    let config: Config = load(next_account_info(accounts)?, program, CONFIG)?;
    let task: Task = load(next_account_info(accounts)?, program, TASK)?;
    let round_account = next_account_info(accounts)?;
    let node: Node = load(next_account_info(accounts)?, program, NODE)?;
    let sysvar = next_account_info(accounts)?;
    let mut round_record: Round = load(round_account, program, ROUND)?;
    require!(task.run == config.run && round_record.run == config.run && round_record.task == task.id, "wrong round");
    require!(node.run == config.run && node.owner == *sender.key, "wrong node record");
    let at = member(&round_record, sender.key)?;
    // Commitments close when sampling begins: the VRF input binds every posted root.
    require!(task.inference_committed && !round_record.sampling && !round_record.votes[at].root_in, "root commitment");
    require!(now()? <= round_record.opened_at + config.params.phase_seconds, "deadline passed");
    signed_by(sysvar, &node.signing_key, &bound_message(&task, round_record.stage, &root))?;
    round_record.votes[at].root = root;
    round_record.votes[at].root_in = true;
    round_record.roots_in += 1;
    store(round_account, &round_record)
}

/// C = Commit(b, r), hiding and binding, posted before the verdict is revealed. The sender's
/// public key is part of the preimage so one verifier cannot replay another's commitment.
///
/// Accounts: verifier (signer), config, round.
pub fn commit_verdict(program: &Pubkey, accounts: Accounts, commitment: [u8; 32]) -> ProgramResult {
    let sender = signer(next_account_info(accounts)?)?;
    let config: Config = load(next_account_info(accounts)?, program, CONFIG)?;
    let round_account = next_account_info(accounts)?;
    let mut round_record: Round = load(round_account, program, ROUND)?;
    require!(round_record.run == config.run, "wrong round");
    let at = member(&round_record, sender.key)?;
    require!(round_record.votes[at].root_in && !round_record.votes[at].verdict_in && !round_record.sampling, "verdict commitment");
    require!(now()? <= round_record.opened_at + config.params.phase_seconds, "deadline passed");
    round_record.votes[at].commitment = commitment;
    round_record.votes[at].verdict_in = true;
    round_record.verdicts_in += 1;
    store(round_account, &round_record)
}

/// The first of the three sampling steps (Eqs. 22 to 27). It checks the VRF proof, the
/// inferencer's signature on its final root and that root's place in the committed trace, then
/// fixes the sampled indices.
///
/// Accounts: scheduler (signer), config, task, stage, round, the inferencer's node record,
/// instructions sysvar.
pub fn begin_sampling(program: &Pubkey, accounts: Accounts, proof: &[u8; 80], inferencer_root: [u8; 32], trace_path: &[[u8; 32]]) -> ProgramResult {
    let sender = next_account_info(accounts)?;
    let config: Config = load(next_account_info(accounts)?, program, CONFIG)?;
    let task: Task = load(next_account_info(accounts)?, program, TASK)?;
    let stage_record: Stage = load(next_account_info(accounts)?, program, STAGE)?;
    let round_account = next_account_info(accounts)?;
    let node: Node = load(next_account_info(accounts)?, program, NODE)?;
    let sysvar = next_account_info(accounts)?;
    let mut round_record: Round = load(round_account, program, ROUND)?;
    scheduler_only(&config, sender)?;
    require!(task.run == config.run && stage_record.run == config.run && stage_record.task == task.id, "wrong stage");
    require!(
        round_record.run == config.run
            && round_record.task == task.id
            && round_record.stage == stage_record.stage
            && round_record.round == stage_record.round,
        "wrong round"
    );
    require!(node.run == config.run && node.owner == stage_record.inferencer, "wrong node record");
    let params = &config.params;
    require!(!round_record.committee.is_empty() && !round_record.sampling, "sampling");
    require!(
        round_record.verdicts_in as usize == round_record.committee.len() || now()? > round_record.opened_at + params.phase_seconds,
        "commitments still open"
    );

    // Eq. 22 and check (1), Eq. 25: the input binds the task, the stage and every posted root,
    // so the indices cannot be known before the commitments are final.
    let mut alpha = Vec::with_capacity(36 + 32 * round_record.votes.len());
    alpha.extend_from_slice(&task.h);
    alpha.extend_from_slice(&stage_record.stage.to_be_bytes());
    for vote in &round_record.votes {
        alpha.extend_from_slice(&vote.root);
    }
    let beta = vrf::verify(&config.vrf_key, &alpha, proof).ok_or_else(|| fail("VRF proof"))?;

    // Check (2), Eq. 26: origin authenticity of the inferencer's final root.
    signed_by(sysvar, &node.signing_key, &bound_message(&task, stage_record.stage, &inferencer_root)).map_err(|_| fail("inferencer signature"))?;
    // The root must be the one fixed by commit_inference, before any verifier committed.
    let leaf = keccak256(&[&[0], &inferencer_root]);
    require!(merkle(&stage_record.trace_root, leaf, task.steps as u64 - 1, trace_path), "not the committed trace");

    round_record.indices = sample(&beta, params.samples, params.width);
    round_record.sample_beta = beta;
    round_record.inferencer_root = inferencer_root;
    round_record.sampling = true;
    store(round_account, &round_record)?;

    let mut record = vec![event::SAMPLING_STARTED];
    record.extend_from_slice(&beta);
    record.extend_from_slice(&(round_record.indices.len() as u32).to_le_bytes());
    for index in &round_record.indices {
        record.extend_from_slice(&index.to_le_bytes());
    }
    sol_log_data(&[&record]);
    Ok(())
}

/// Check (3), Eq. 27: inclusion of the reference values at their sampled indices, a few per
/// transaction and in any order.
///
/// Accounts: scheduler (signer), config, round.
pub fn sampling_openings(program: &Pubkey, accounts: Accounts, start: u32, values: &[u8], paths: &[[u8; 32]]) -> ProgramResult {
    let sender = next_account_info(accounts)?;
    let config: Config = load(next_account_info(accounts)?, program, CONFIG)?;
    let round_account = next_account_info(accounts)?;
    let mut round_record: Round = load(round_account, program, ROUND)?;
    scheduler_only(&config, sender)?;
    require!(round_record.run == config.run, "wrong round");
    require!(round_record.sampling && !round_record.sampled, "sampling");
    let held = check_openings(&round_record.inferencer_root, &round_record.indices, start, values, paths)?;
    for (j, ok) in held.into_iter().enumerate() {
        let at = start as usize + j;
        require!(ok, "reference openings");
        require!(round_record.reference_opened[at] == 0, "opening already given");
        round_record.reference_opened[at] = 1;
        round_record.reference_count += 1;
        round_record.reference_values[4 * at..4 * at + 4].copy_from_slice(&values[4 * j..4 * j + 4]);
    }
    store(round_account, &round_record)
}

/// Closes the sampling step once every reference value has been opened. The reveal phase
/// starts here.
///
/// Accounts: scheduler (signer), config, round.
pub fn finish_sampling(program: &Pubkey, accounts: Accounts) -> ProgramResult {
    let sender = next_account_info(accounts)?;
    let config: Config = load(next_account_info(accounts)?, program, CONFIG)?;
    let round_account = next_account_info(accounts)?;
    let mut round_record: Round = load(round_account, program, ROUND)?;
    scheduler_only(&config, sender)?;
    require!(round_record.run == config.run, "wrong round");
    require!(round_record.sampling && !round_record.sampled, "sampling");
    require!(round_record.reference_count == config.params.samples, "reference openings");
    round_record.sampled = true;
    round_record.sampled_at = now()?;
    store(round_account, &round_record)?;
    sol_log_data(&[&[event::SAMPLING_GENERATED], &round_record.inferencer_root, &round_record.reference_values]);
    Ok(())
}

/// The first of the three steps of a verifier's reveal (Eqs. 28 and 29): OpenCommit(C; b, r).
///
/// Accounts: verifier (signer), config, round.
pub fn begin_reveal(program: &Pubkey, accounts: Accounts, verdict: bool, randomness: &[u8; 32]) -> ProgramResult {
    let sender = signer(next_account_info(accounts)?)?;
    let config: Config = load(next_account_info(accounts)?, program, CONFIG)?;
    let round_account = next_account_info(accounts)?;
    let mut round_record: Round = load(round_account, program, ROUND)?;
    require!(round_record.run == config.run, "wrong round");
    let at = member(&round_record, sender.key)?;
    let vote = &mut round_record.votes[at];
    require!(round_record.sampled && vote.verdict_in && !vote.opening && !vote.revealed && !round_record.adjudicated, "reveal");
    require!(now()? <= round_record.sampled_at + config.params.phase_seconds, "deadline passed");
    require!(keccak256(&[&[verdict as u8], randomness, sender.key.as_ref()]) == vote.commitment, "commitment opening");
    vote.opening = true;
    vote.verdict = verdict;
    store(round_account, &round_record)
}

/// The second step: Merkle inclusion against the verifier's own root, a few samples per
/// transaction and in any order. An opening that does not hold is recorded, not refused: the
/// reveal then closes as invalid.
///
/// Accounts: verifier (signer), config, round.
pub fn reveal_openings(program: &Pubkey, accounts: Accounts, start: u32, values: &[u8], paths: &[[u8; 32]]) -> ProgramResult {
    let sender = signer(next_account_info(accounts)?)?;
    let config: Config = load(next_account_info(accounts)?, program, CONFIG)?;
    let round_account = next_account_info(accounts)?;
    let mut round_record: Round = load(round_account, program, ROUND)?;
    require!(round_record.run == config.run, "wrong round");
    let at = member(&round_record, sender.key)?;
    require!(round_record.votes[at].opening && !round_record.votes[at].revealed && !round_record.adjudicated, "reveal");
    require!(now()? <= round_record.sampled_at + config.params.phase_seconds, "deadline passed");
    let held = check_openings(&round_record.votes[at].root, &round_record.indices, start, values, paths)?;
    let vote = &mut round_record.votes[at];
    for (j, ok) in held.into_iter().enumerate() {
        let sample = start as usize + j;
        require!(vote.opened[sample] == 0, "opening already given");
        vote.opened[sample] = 1;
        vote.opened_count += 1;
        if ok {
            vote.values[4 * sample..4 * sample + 4].copy_from_slice(&values[4 * j..4 * j + 4]);
        } else {
            vote.failed = true;
        }
    }
    store(round_account, &round_record)
}

/// The third step: value matching and the agreement rate against the inferencer's reference.
/// Closes the reveal: it is valid when all n openings arrived and held.
///
/// Accounts: verifier (signer), config, round.
pub fn finish_reveal(program: &Pubkey, accounts: Accounts) -> ProgramResult {
    let sender = signer(next_account_info(accounts)?)?;
    let config: Config = load(next_account_info(accounts)?, program, CONFIG)?;
    let round_account = next_account_info(accounts)?;
    let mut round_record: Round = load(round_account, program, ROUND)?;
    require!(round_record.run == config.run, "wrong round");
    let at = member(&round_record, sender.key)?;
    require!(round_record.votes[at].opening && !round_record.votes[at].revealed && !round_record.adjudicated, "reveal");
    require!(now()? <= round_record.sampled_at + config.params.phase_seconds, "deadline passed");
    let valid = !round_record.votes[at].failed && round_record.votes[at].opened_count == config.params.samples;
    let (same, agreeing) =
        if valid { consistent(&config.params, &round_record.reference_values, &round_record.votes[at].values) } else { (false, 0) };
    let vote = &mut round_record.votes[at];
    vote.revealed = true;
    vote.valid = valid;
    vote.consistent = same;
    vote.agreeing = agreeing;
    let verdict = vote.verdict;
    round_record.reveals += 1;
    store(round_account, &round_record)?;
    sol_log_data(&[&[event::VERIFICATION_REVEALED, verdict as u8, valid as u8], &agreeing.to_le_bytes(), &[same as u8]]);
    Ok(())
}
