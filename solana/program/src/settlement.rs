//! Rewards and penalties for one stage (Appendix E.3).

use crate::accounts::{event, fail, load, now, require, store, Accounts};
use crate::state::{Config, Node, Round, Stage, ACCEPTED, CONFIG, INDETERMINATE, NODE, REJECTED, ROUND, STAGE};
use solana_program::{
    account_info::{next_account_info, AccountInfo},
    entrypoint::ProgramResult,
    log::sol_log_data,
    program_error::ProgramError,
    pubkey::Pubkey,
};

/// The reward pool and the node records of one settlement, changed in memory and stored once.
struct Ledger<'a, 'b> {
    pool: u64,
    unit: u64,
    nodes: Vec<(&'a AccountInfo<'b>, Node)>,
}

impl Ledger<'_, '_> {
    fn stake(&mut self, who: &Pubkey) -> Result<&mut u64, ProgramError> {
        self.nodes.iter_mut().find(|(_, node)| node.owner == *who).map(|(_, node)| &mut node.stake).ok_or_else(|| fail("node record missing"))
    }

    /// Moves `units` from the pool to the stake of `who`, or back when negative, up to the stake held.
    fn pay(&mut self, who: &Pubkey, units: i64) -> ProgramResult {
        if units >= 0 {
            let amount = units as u64 * self.unit;
            require!(self.pool >= amount, "reward pool empty");
            self.pool -= amount;
            *self.stake(who)? += amount;
        } else {
            let due = (-units) as u64 * self.unit;
            let stake = self.stake(who)?;
            let amount = due.min(*stake);
            *stake -= amount;
            self.pool += amount;
        }
        sol_log_data(&[&[event::SETTLED], who.as_ref(), &units.to_le_bytes()]);
        Ok(())
    }
}

/// Accounts: anyone (signer), config, stage, first round, the second round if one was requested,
/// then the node record of every party to be paid or slashed (the inferencer and all committee
/// members), in any order.
pub fn settle(program: &Pubkey, accounts: Accounts) -> ProgramResult {
    let _sender = next_account_info(accounts)?;
    let config_account = next_account_info(accounts)?;
    let stage_account = next_account_info(accounts)?;
    let mut config: Config = load(config_account, program, CONFIG)?;
    let mut stage_record: Stage = load(stage_account, program, STAGE)?;
    let first: Round = load(next_account_info(accounts)?, program, ROUND)?;
    require!(stage_record.run == config.run && first.run == config.run, "wrong stage");
    require!(first.task == stage_record.task && first.stage == stage_record.stage && first.round == 1, "wrong round");
    require!(first.adjudicated && !stage_record.settled, "settlement");
    let mut second: Option<Round> = None;
    let final_outcome = if stage_record.reverify_requested {
        require!(stage_record.round == 2, "second round pending");
        let round_record: Round = load(next_account_info(accounts)?, program, ROUND)?;
        require!(
            round_record.run == config.run
                && round_record.task == stage_record.task
                && round_record.stage == stage_record.stage
                && round_record.round == 2,
            "wrong round"
        );
        require!(round_record.adjudicated, "second round pending");
        let outcome = round_record.outcome;
        second = Some(round_record);
        outcome
    } else {
        require!(first.outcome == ACCEPTED || now()? > stage_record.dispute_deadline, "dispute window open");
        first.outcome
    };
    stage_record.settled = true;
    let reward = config.params.reward as i64;
    let slash = -(config.params.slash as i64);

    let mut ledger = Ledger { pool: config.pool, unit: config.params.unit, nodes: Vec::new() };
    for account in accounts {
        let node: Node = load(account, program, NODE)?;
        require!(node.run == config.run, "wrong node record");
        ledger.nodes.push((account, node));
    }

    if let Some(second) = &second {
        // A successful dispute refunds the cost; a failed one forfeits it to the pool.
        if stage_record.dispute_paid {
            let cost = config.params.dispute_cost as u64 * config.params.unit;
            if final_outcome == REJECTED {
                ledger.pool += cost;
                sol_log_data(&[&[event::SETTLED], stage_record.inferencer.as_ref(), &(-(config.params.dispute_cost as i64)).to_le_bytes()]);
            } else {
                *ledger.stake(&stage_record.inferencer)? += cost;
            }
        }
        for (who, vote) in second.committee.iter().zip(&second.votes) {
            if final_outcome == INDETERMINATE {
                continue;
            }
            ledger.pay(who, if vote.agrees { reward } else { slash })?;
        }
    }
    if final_outcome == ACCEPTED {
        ledger.pay(&stage_record.inferencer, reward)?;
    }
    if final_outcome == REJECTED {
        ledger.pay(&stage_record.inferencer, slash)?;
    }

    for (who, vote) in first.committee.iter().zip(&first.votes) {
        let good = if stage_record.reverify_requested && final_outcome == ACCEPTED {
            // The penalty is rescinded and every first-round verifier who voted False is slashed.
            vote.valid && vote.verdict
        } else if final_outcome == INDETERMINATE {
            continue;
        } else {
            vote.agrees
        };
        ledger.pay(who, if good { reward } else { slash })?;
    }

    config.pool = ledger.pool;
    for (account, node) in &ledger.nodes {
        store(account, node)?;
    }
    store(stage_account, &stage_record)?;
    store(config_account, &config)
}
