//! Adjudication of a round and the inferencer's dispute (Section 5.7, Appendix E.2).

use crate::accounts::{event, load, now, pay_in, require, signer, store, Accounts};
use crate::math::consistent;
use crate::state::{Config, Round, Stage, ACCEPTED, CONFIG, INDETERMINATE, REJECTED, ROUND, STAGE};
use solana_program::{account_info::next_account_info, entrypoint::ProgramResult, log::sol_log_data, pubkey::Pubkey};

/// The representative of the cluster that holds entry `i` of a union-find forest.
fn find(parent: &[usize], mut i: usize) -> usize {
    while parent[i] != i {
        i = parent[i];
    }
    i
}

/// Verifiers whose revealed samples agree pairwise are joined into clusters (Kruskal stopped at
/// the tolerance gives the connected components of the agreement graph). A cluster holding a
/// quorum q = ceil(m / 2) is the proper cluster. The inference is accepted when that cluster
/// votes to accept and the inferencer's samples agree with one of its members. A consistent
/// cluster that votes to reject makes the negative verdict stand. No proper cluster: no consensus.
///
/// Accounts: anyone (signer), config, stage, round.
pub fn adjudicate(program: &Pubkey, accounts: Accounts) -> ProgramResult {
    let _sender = next_account_info(accounts)?;
    let config: Config = load(next_account_info(accounts)?, program, CONFIG)?;
    let stage_account = next_account_info(accounts)?;
    let round_account = next_account_info(accounts)?;
    let mut stage_record: Stage = load(stage_account, program, STAGE)?;
    let mut round_record: Round = load(round_account, program, ROUND)?;
    require!(stage_record.run == config.run && round_record.run == config.run, "wrong stage");
    require!(
        round_record.task == stage_record.task && round_record.stage == stage_record.stage && round_record.round == stage_record.round,
        "wrong round"
    );
    let params = &config.params;
    require!(round_record.sampled && !round_record.adjudicated, "adjudication");
    require!(
        round_record.reveals as usize == round_record.committee.len() || now()? > round_record.sampled_at + params.phase_seconds,
        "reveals still open"
    );
    let votes = &mut round_record.votes;
    let size = votes.len();

    // Union-find over the verifiers whose openings held: an edge joins two whose samples agree.
    let mut parent: Vec<usize> = (0..size).collect();
    for i in 0..size {
        if !votes[i].valid {
            continue;
        }
        for j in i + 1..size {
            if !votes[j].valid || find(&parent, i) == find(&parent, j) {
                continue;
            }
            if consistent(params, &votes[i].values, &votes[j].values).0 {
                let (from, to) = (find(&parent, j), find(&parent, i));
                parent[from] = to;
            }
        }
    }

    // The cluster holding a quorum q = ceil(m / 2), if there is one. There can be at most one.
    let mut proper: Option<(usize, u32)> = None;
    for i in 0..size {
        if !votes[i].valid || find(&parent, i) != i {
            continue;
        }
        let members = (0..size).filter(|&j| votes[j].valid && find(&parent, j) == i).count();
        if members >= size.div_ceil(2) {
            proper = Some((i, members as u32));
            break;
        }
    }

    let mut outcome = INDETERMINATE;
    let mut cluster_verdict = false;
    let mut cluster_size = 0;
    if let Some((root, members)) = proper {
        cluster_size = members;
        let mut accepting = 0u32;
        let mut inferencer_near = false;
        for (i, vote) in votes.iter().enumerate() {
            if !vote.valid || find(&parent, i) != root {
                continue;
            }
            if vote.verdict {
                accepting += 1;
            }
            if !inferencer_near {
                inferencer_near = consistent(params, &round_record.reference_values, &vote.values).0;
            }
        }
        cluster_verdict = 2 * accepting > members;
        outcome = if cluster_verdict && inferencer_near { ACCEPTED } else { REJECTED };
        for (i, vote) in votes.iter_mut().enumerate() {
            vote.agrees = vote.valid && find(&parent, i) == root && vote.verdict == (outcome == ACCEPTED);
        }
    }

    round_record.adjudicated = true;
    round_record.outcome = outcome;
    round_record.cluster_verdict = cluster_verdict;
    round_record.cluster_size = cluster_size;
    if round_record.round == 1 {
        if outcome == REJECTED {
            stage_record.dispute_deadline = now()? + params.dispute_seconds;
        }
        if outcome == INDETERMINATE {
            stage_record.reverify_requested = true;
        }
    }
    store(stage_account, &stage_record)?;
    store(round_account, &round_record)?;
    sol_log_data(&[&[event::ADJUDICATED, round_record.round, outcome], &cluster_size.to_le_bytes(), &[cluster_verdict as u8]]);
    Ok(())
}

/// Dispute resolution (Section 5.7): a rejected inferencer may pay for a second verification by
/// a larger committee, at most once.
///
/// Accounts: inferencer (signer), config, stage, first round, system program.
pub fn dispute(program: &Pubkey, accounts: Accounts) -> ProgramResult {
    let sender = signer(next_account_info(accounts)?)?;
    let config_account = next_account_info(accounts)?;
    let stage_account = next_account_info(accounts)?;
    let round_record: Round = load(next_account_info(accounts)?, program, ROUND)?;
    let system = next_account_info(accounts)?;
    let config: Config = load(config_account, program, CONFIG)?;
    let mut stage_record: Stage = load(stage_account, program, STAGE)?;
    require!(stage_record.run == config.run && round_record.run == config.run, "wrong stage");
    require!(round_record.task == stage_record.task && round_record.stage == stage_record.stage && round_record.round == 1, "wrong round");
    require!(*sender.key == stage_record.inferencer && stage_record.round == 1 && round_record.outcome == REJECTED, "nothing to dispute");
    require!(!stage_record.reverify_requested && now()? <= stage_record.dispute_deadline, "dispute window");
    let cost = config.params.dispute_cost as u64 * config.params.unit;
    pay_in(sender, config_account, system, cost)?;
    stage_record.reverify_requested = true;
    stage_record.dispute_paid = true;
    store(stage_account, &stage_record)?;
    sol_log_data(&[&[event::DISPUTED], sender.key.as_ref(), &cost.to_le_bytes()]);
    Ok(())
}
