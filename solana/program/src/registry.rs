//! Registration of a run, its scheduler and its nodes (Section 5.2).

use crate::accounts::{create, load, now, pay_in, require, signer, store, store_resized, Accounts};
use crate::state::{Config, Group, Node, Params, CONFIG, GROUP, NODE};
use solana_program::{account_info::next_account_info, entrypoint::ProgramResult, pubkey::Pubkey};

/// Fixes the parameters of one run and funds its reward pool.
///
/// Accounts: operator (signer), config, system program.
pub fn initialize(program: &Pubkey, accounts: Accounts, run: [u8; 8], params: Params, pool: u64) -> ProgramResult {
    let operator = signer(next_account_info(accounts)?)?;
    let config_account = next_account_info(accounts)?;
    let system = next_account_info(accounts)?;
    require!(params.verifiers % 2 == 1 && params.reverifiers % 2 == 1 && params.reverifiers > params.verifiers, "committee sizes");
    require!(params.samples > 0 && params.samples <= params.width, "sample size");
    let config = Config {
        kind: CONFIG,
        run,
        params,
        scheduler: Pubkey::default(),
        vrf_key: [0; 32],
        scheduler_signing_key: [0; 32],
        scheduler_stake: 0,
        pool,
        tasks: 0,
        scheduler_endpoint: String::new(),
    };
    create(program, operator, config_account, system, &[b"config", &run], &config)?;
    pay_in(operator, config_account, system, pool)
}

/// Accounts: scheduler (signer), config, system program.
pub fn register_scheduler(program: &Pubkey, accounts: Accounts, vrf_key: [u8; 32], signing_key: [u8; 32], endpoint: String) -> ProgramResult {
    let sender = signer(next_account_info(accounts)?)?;
    let config_account = next_account_info(accounts)?;
    let system = next_account_info(accounts)?;
    let mut config: Config = load(config_account, program, CONFIG)?;
    require!(config.scheduler == Pubkey::default(), "scheduler registered");
    pay_in(sender, config_account, system, config.params.scheduler_stake)?;
    config.scheduler = *sender.key;
    config.vrf_key = vrf_key;
    config.scheduler_signing_key = signing_key;
    config.scheduler_endpoint = endpoint;
    config.scheduler_stake = config.params.scheduler_stake;
    store_resized(sender, config_account, system, &config)
}

/// Accounts: node (signer), config, node record, group, system program.
pub fn register_node(
    program: &Pubkey,
    accounts: Accounts,
    model_id: [u8; 32],
    parameter_hash: [u8; 32],
    slice: u32,
    signing_key: [u8; 32],
    endpoint: String,
) -> ProgramResult {
    let sender = signer(next_account_info(accounts)?)?;
    let config_account = next_account_info(accounts)?;
    let node_account = next_account_info(accounts)?;
    let group_account = next_account_info(accounts)?;
    let system = next_account_info(accounts)?;
    let config: Config = load(config_account, program, CONFIG)?;
    let node = Node {
        kind: NODE,
        run: config.run,
        owner: *sender.key,
        model_id,
        parameter_hash,
        signing_key,
        slice,
        exit_at: 0,
        active: true,
        stake: config.params.node_stake,
        endpoint,
    };
    if node_account.data_is_empty() {
        create(program, sender, node_account, system, &[b"node", &config.run, sender.key.as_ref()], &node)?;
    } else {
        let old: Node = load(node_account, program, NODE)?;
        require!(old.run == config.run && old.owner == *sender.key, "wrong node record");
        require!(!old.active && old.stake == 0, "already registered");
        store_resized(sender, node_account, system, &node)?;
    }
    pay_in(sender, config_account, system, config.params.node_stake)?;

    // Members are kept in registration order: the role permutation indexes into this list.
    let slice_bytes = slice.to_le_bytes();
    if group_account.data_is_empty() {
        let group = Group { kind: GROUP, run: config.run, model_id, slice, members: vec![*sender.key] };
        create(program, sender, group_account, system, &[b"group", &config.run, &model_id, &slice_bytes], &group)
    } else {
        let mut group: Group = load(group_account, program, GROUP)?;
        require!(group.run == config.run && group.model_id == model_id && group.slice == slice, "wrong group");
        group.members.push(*sender.key);
        store_resized(sender, group_account, system, &group)
    }
}

/// Withdrawal waits out the unbonding period so pending verifications can still slash.
///
/// Accounts: node (signer), config, node record, group.
pub fn deregister(program: &Pubkey, accounts: Accounts) -> ProgramResult {
    let sender = signer(next_account_info(accounts)?)?;
    let config: Config = load(next_account_info(accounts)?, program, CONFIG)?;
    let node_account = next_account_info(accounts)?;
    let group_account = next_account_info(accounts)?;
    let mut node: Node = load(node_account, program, NODE)?;
    require!(node.run == config.run && node.owner == *sender.key, "wrong node record");
    require!(node.active, "not registered");
    node.active = false;
    node.exit_at = now()? + config.params.unbonding_seconds;
    let mut group: Group = load(group_account, program, GROUP)?;
    require!(group.run == config.run && group.model_id == node.model_id && group.slice == node.slice, "wrong group");
    if let Some(at) = group.members.iter().position(|key| key == sender.key) {
        group.members.swap_remove(at);
    }
    // The group account keeps its size; the list inside it is one entry shorter.
    store(group_account, &group)?;
    store(node_account, &node)
}

/// Accounts: node (signer), config, node record.
pub fn withdraw(program: &Pubkey, accounts: Accounts) -> ProgramResult {
    let sender = signer(next_account_info(accounts)?)?;
    let config_account = next_account_info(accounts)?;
    let node_account = next_account_info(accounts)?;
    let config: Config = load(config_account, program, CONFIG)?;
    let mut node: Node = load(node_account, program, NODE)?;
    require!(node.run == config.run && node.owner == *sender.key, "wrong node record");
    require!(!node.active && node.exit_at != 0 && now()? >= node.exit_at, "unbonding");
    let amount = node.stake;
    node.stake = 0;
    store(node_account, &node)?;
    **config_account.try_borrow_mut_lamports()? -= amount;
    **sender.try_borrow_mut_lamports()? += amount;
    Ok(())
}
