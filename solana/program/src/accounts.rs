//! Reading, writing and creating the program's accounts, and the checks every instruction shares.

use crate::state::{Config, Round};
use borsh::{BorshDeserialize, BorshSerialize};
use solana_program::{
    account_info::AccountInfo,
    clock::Clock,
    entrypoint::ProgramResult,
    msg,
    program::{invoke, invoke_signed},
    program_error::ProgramError,
    pubkey::Pubkey,
    rent::Rent,
    system_instruction, system_program,
    sysvar::Sysvar,
};

/// The accounts of one instruction, taken in order.
pub type Accounts<'a, 'b> = &'a mut std::slice::Iter<'b, AccountInfo<'b>>;

/// Event tags, the first byte of each logged record.
pub mod event {
    pub const TASK_CREATED: u8 = 1;
    pub const ROLES_ASSIGNED: u8 = 2;
    pub const SAMPLING_STARTED: u8 = 3;
    pub const VERIFICATION_REVEALED: u8 = 4;
    pub const ADJUDICATED: u8 = 5;
    pub const SETTLED: u8 = 6;
    pub const VRF_CHECKED: u8 = 7;
    pub const SAMPLING_GENERATED: u8 = 8;
    pub const DISPUTED: u8 = 9;
}

pub fn fail(reason: &str) -> ProgramError {
    msg!("VeriLLM: {}", reason);
    ProgramError::Custom(1)
}

macro_rules! require {
    ($condition:expr, $reason:expr) => {
        if !($condition) {
            return Err($crate::accounts::fail($reason));
        }
    };
}
pub(crate) use require;

pub fn load<T: BorshDeserialize>(account: &AccountInfo, program: &Pubkey, kind: u8) -> Result<T, ProgramError> {
    require!(account.owner == program, "account not owned by the program");
    let data = account.try_borrow_data()?;
    require!(!data.is_empty() && data[0] == kind, "wrong kind of account");
    T::deserialize(&mut &data[..]).map_err(|_| fail("account does not decode"))
}

pub fn store<T: BorshSerialize>(account: &AccountInfo, value: &T) -> ProgramResult {
    let mut data = account.try_borrow_mut_data()?;
    value.serialize(&mut &mut data[..]).map_err(|_| fail("account does not encode"))
}

fn size_of<T: BorshSerialize>(value: &T) -> Result<usize, ProgramError> {
    borsh::object_length(value).map_err(|_| fail("account does not encode"))
}

pub fn pay_in<'a>(from: &AccountInfo<'a>, to: &AccountInfo<'a>, system: &AccountInfo<'a>, lamports: u64) -> ProgramResult {
    require!(*system.key == system_program::ID, "system program");
    invoke(&system_instruction::transfer(from.key, to.key, lamports), &[from.clone(), to.clone(), system.clone()])
}

/// Creates the program-derived account at `seeds` and writes `value` into it.
pub fn create<'a, T: BorshSerialize>(
    program: &Pubkey,
    payer: &AccountInfo<'a>,
    target: &AccountInfo<'a>,
    system: &AccountInfo<'a>,
    seeds: &[&[u8]],
    value: &T,
) -> ProgramResult {
    require!(*system.key == system_program::ID, "system program");
    let (address, bump) = Pubkey::find_program_address(seeds, program);
    require!(address == *target.key, "wrong account address");
    require!(target.data_is_empty() && *target.owner == system_program::ID, "account exists");
    let size = size_of(value)?;
    let bump = [bump];
    let mut signer_seeds = seeds.to_vec();
    signer_seeds.push(&bump);
    invoke_signed(
        &system_instruction::create_account(payer.key, target.key, Rent::get()?.minimum_balance(size), size as u64, program),
        &[payer.clone(), target.clone(), system.clone()],
        &[&signer_seeds],
    )?;
    store(target, value)
}

/// Resizes an account to fit `value`, tops up its rent from `payer` and writes `value`.
pub fn store_resized<'a, T: BorshSerialize>(payer: &AccountInfo<'a>, target: &AccountInfo<'a>, system: &AccountInfo<'a>, value: &T) -> ProgramResult {
    let size = size_of(value)?;
    if size != target.data_len() {
        target.realloc(size, false)?;
        let due = Rent::get()?.minimum_balance(size).saturating_sub(target.lamports());
        if due > 0 {
            pay_in(payer, target, system, due)?;
        }
    }
    store(target, value)
}

pub fn now() -> Result<i64, ProgramError> {
    Ok(Clock::get()?.unix_timestamp)
}

pub fn signer<'a, 'b>(account: &'b AccountInfo<'a>) -> Result<&'b AccountInfo<'a>, ProgramError> {
    require!(account.is_signer, "signature missing");
    Ok(account)
}

pub fn scheduler_only(config: &Config, sender: &AccountInfo) -> ProgramResult {
    require!(sender.is_signer && config.scheduler != Pubkey::default() && config.scheduler == *sender.key, "scheduler only");
    Ok(())
}

/// The position of `who` on the round's committee.
pub fn member(round_record: &Round, who: &Pubkey) -> Result<usize, ProgramError> {
    round_record.committee.iter().position(|key| key == who).ok_or_else(|| fail("not on the committee"))
}
