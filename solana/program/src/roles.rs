//! Tasks, role assignment and the inference commitment (Sections 5.3 and 5.4).

use crate::accounts::{create, event, fail, load, now, require, scheduler_only, store, Accounts};
use crate::math::permutation;
use crate::state::{Config, Group, Round, Stage, Task, Vote, CONFIG, GROUP, NONE, ROUND, STAGE, TASK};
use crate::vrf;
use solana_program::{account_info::next_account_info, entrypoint::ProgramResult, log::sol_log_data, pubkey::Pubkey};

/// Accounts: scheduler (signer), config, task, system program.
pub fn create_task(program: &Pubkey, accounts: Accounts, h: [u8; 32], model_id: [u8; 32], stages: u32) -> ProgramResult {
    let sender = next_account_info(accounts)?;
    let config_account = next_account_info(accounts)?;
    let task_account = next_account_info(accounts)?;
    let system = next_account_info(accounts)?;
    let mut config: Config = load(config_account, program, CONFIG)?;
    scheduler_only(&config, sender)?;
    let id = config.tasks;
    let task = Task { kind: TASK, run: config.run, id, h, model_id, output_hash: [0; 32], stages, steps: 0, inference_committed: false };
    create(program, sender, task_account, system, &[b"task", &config.run, &id.to_le_bytes()], &task)?;
    config.tasks += 1;
    store(config_account, &config)?;
    sol_log_data(&[&[event::TASK_CREATED], &id.to_le_bytes()]);
    Ok(())
}

/// Eqs. 8 to 10. The VRF input is <h, i, L_i>; the output seeds a Fisher-Yates permutation of the
/// group, whose first entry is the inferencer and next m entries the verifiers. A second round
/// (dispute or no consensus) appends the round number to the input and draws m' verifiers.
///
/// Accounts: scheduler (signer), config, task, group, stage, round, system program.
pub fn assign_roles(program: &Pubkey, accounts: Accounts, stage: u32, proof: &[u8; 80]) -> ProgramResult {
    let sender = next_account_info(accounts)?;
    let config: Config = load(next_account_info(accounts)?, program, CONFIG)?;
    let task: Task = load(next_account_info(accounts)?, program, TASK)?;
    let group_account = next_account_info(accounts)?;
    let stage_account = next_account_info(accounts)?;
    let round_account = next_account_info(accounts)?;
    let system = next_account_info(accounts)?;
    scheduler_only(&config, sender)?;
    require!(task.run == config.run, "wrong task");
    require!(stage >= 1 && stage <= task.stages, "stage");
    require!(!group_account.data_is_empty(), "group too small");
    let group: Group = load(group_account, program, GROUP)?;
    require!(group.run == config.run && group.model_id == task.model_id && group.slice == stage, "wrong group");
    let members = &group.members;
    let size = members.len() as u32;
    let params = &config.params;

    let existing: Option<Stage> = if stage_account.data_is_empty() { None } else { Some(load(stage_account, program, STAGE)?) };
    let mut alpha = Vec::with_capacity(44);
    alpha.extend_from_slice(&task.h);
    alpha.extend_from_slice(&stage.to_be_bytes());
    alpha.extend_from_slice(&size.to_be_bytes());
    let round: u8 = match &existing {
        None => 1,
        Some(stage_record) => {
            require!(stage_record.run == config.run && stage_record.task == task.id && stage_record.stage == stage, "wrong stage");
            require!(stage_record.round == 1 && stage_record.reverify_requested, "no second round due");
            alpha.extend_from_slice(&2u32.to_be_bytes());
            2
        }
    };
    let beta = vrf::verify(&config.vrf_key, &alpha, proof).ok_or_else(|| fail("VRF proof"))?;
    let drawn = permutation(&beta, size).into_iter().map(|index| members[index as usize]);

    let mut opened_at = 0;
    let (stage_record, committee): (Stage, Vec<Pubkey>) = match existing {
        None => {
            require!(size > params.verifiers, "group too small");
            let drawn: Vec<Pubkey> = drawn.take(1 + params.verifiers as usize).collect();
            let stage_record = Stage {
                kind: STAGE,
                run: config.run,
                task: task.id,
                stage,
                inferencer: drawn[0],
                trace_root: [0; 32],
                round,
                reverify_requested: false,
                dispute_paid: false,
                settled: false,
                dispute_deadline: 0,
            };
            (stage_record, drawn[1..].to_vec())
        }
        Some(mut stage_record) => {
            require!(size > params.reverifiers, "group too small");
            let committee = drawn.filter(|key| *key != stage_record.inferencer).take(params.reverifiers as usize).collect();
            opened_at = now()?;
            stage_record.round = round;
            (stage_record, committee)
        }
    };

    // Every list in the round has its final length from the start, so the account never grows.
    let samples = params.samples as usize;
    let votes = committee.iter().map(|_| Vote { values: vec![0; 4 * samples], opened: vec![0; samples], ..Vote::default() }).collect();
    let round_record = Round {
        kind: ROUND,
        run: config.run,
        task: task.id,
        stage,
        round,
        committee,
        role_beta: beta,
        sample_beta: [0; 64],
        inferencer_root: [0; 32],
        indices: vec![0; samples],
        reference_values: vec![0; 4 * samples],
        reference_opened: vec![0; samples],
        reference_count: 0,
        opened_at,
        sampled_at: 0,
        roots_in: 0,
        verdicts_in: 0,
        reveals: 0,
        sampling: false,
        sampled: false,
        adjudicated: false,
        cluster_verdict: false,
        outcome: NONE,
        cluster_size: 0,
        votes,
    };
    let (task_bytes, stage_bytes) = (task.id.to_le_bytes(), stage.to_le_bytes());
    if round == 1 {
        create(program, sender, stage_account, system, &[b"stage", &config.run, &task_bytes, &stage_bytes], &stage_record)?;
    } else {
        store(stage_account, &stage_record)?;
    }
    create(program, sender, round_account, system, &[b"round", &config.run, &task_bytes, &stage_bytes, &[round]], &round_record)?;

    let mut record = vec![event::ROLES_ASSIGNED, round];
    record.extend_from_slice(&beta);
    record.extend_from_slice(stage_record.inferencer.as_ref());
    record.extend_from_slice(&(round_record.committee.len() as u32).to_le_bytes());
    for key in &round_record.committee {
        record.extend_from_slice(key.as_ref());
    }
    sol_log_data(&[&record]);
    Ok(())
}

/// "The scheduler also commits job metadata and Merkle roots of intermediate states" (Section 5.1).
/// One root per segment, over that segment's per-step roots, binds the whole trace in 32 bytes.
///
/// Accounts: scheduler (signer), config, task, then the stage and first-round accounts of each
/// stage in order.
pub fn commit_inference(program: &Pubkey, accounts: Accounts, steps: u32, output_hash: [u8; 32], trace_roots: &[[u8; 32]]) -> ProgramResult {
    let sender = next_account_info(accounts)?;
    let config: Config = load(next_account_info(accounts)?, program, CONFIG)?;
    let task_account = next_account_info(accounts)?;
    let mut task: Task = load(task_account, program, TASK)?;
    scheduler_only(&config, sender)?;
    require!(task.run == config.run, "wrong task");
    require!(!task.inference_committed && trace_roots.len() == task.stages as usize && steps >= 2, "inference commitment");
    let opened_at = now()?;
    for stage in 1..=task.stages {
        let stage_account = next_account_info(accounts)?;
        let round_account = next_account_info(accounts)?;
        require!(!stage_account.data_is_empty() && !round_account.data_is_empty(), "roles not assigned");
        let mut stage_record: Stage = load(stage_account, program, STAGE)?;
        let mut round_record: Round = load(round_account, program, ROUND)?;
        require!(stage_record.run == config.run && stage_record.task == task.id && stage_record.stage == stage, "wrong stage");
        require!(
            round_record.run == config.run && round_record.task == task.id && round_record.stage == stage && round_record.round == 1,
            "wrong round"
        );
        require!(stage_record.round == 1, "roles not assigned");
        stage_record.trace_root = trace_roots[stage as usize - 1];
        round_record.opened_at = opened_at;
        store(stage_account, &stage_record)?;
        store(round_account, &round_record)?;
    }
    task.steps = steps;
    task.output_hash = output_hash;
    task.inference_committed = true;
    store(task_account, &task)
}
