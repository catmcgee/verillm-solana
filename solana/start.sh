#!/usr/bin/env bash
# Start everything from scratch: local keys, the program build, a local test validator and
# the deployment. Safe to run again: it rebuilds and redeploys, and leaves a running
# validator alone.
#
# Nothing here touches the machine's Solana CLI configuration, default keypair or installed
# release. Every command names the local validator and a keypair made here under .keys/.
#
# The validator and CLI come from an Agave release unpacked into .agave/. The demo sends v1
# transactions of up to 4096 bytes (SIMD-0385); Agave 4.1 already runs them, but its RPC
# still refuses anything over 1232 bytes, so a newer release is needed.
set -euo pipefail
cd "$(dirname "$0")"

AGAVE_VERSION="${VERILLM_AGAVE_VERSION:-v4.4.1}"
PORT="${VERILLM_RPC_PORT:-8899}"
URL="http://127.0.0.1:$PORT"
LEDGER="${VERILLM_LEDGER:-/tmp/verillm-ledger}"
KEYS=.keys
# Set VERILLM_AGAVE_BIN to the bin directory of another Agave install (4.2 or later) to use
# that instead of fetching one.
BIN="${VERILLM_AGAVE_BIN:-$PWD/.agave/solana-release/bin}"

if [ -n "${VERILLM_AGAVE_BIN:-}" ]; then
  echo "using Agave in $BIN"
elif [ ! -x "$BIN/solana-test-validator" ] || ! "$BIN/solana-test-validator" --version | grep -q " ${AGAVE_VERSION#v} "; then
  echo "fetching Agave $AGAVE_VERSION into .agave/"
  rm -rf .agave && mkdir -p .agave
  curl -sSfL "https://github.com/anza-xyz/agave/releases/download/$AGAVE_VERSION/solana-release-x86_64-unknown-linux-gnu.tar.bz2" | tar xj -C .agave
fi

mkdir -p "$KEYS"
for name in payer program; do
  [ -f "$KEYS/$name.json" ] || "$BIN/solana-keygen" new --no-bip39-passphrase --silent --outfile "$KEYS/$name.json" >/dev/null
done
PAYER="$("$BIN/solana-keygen" pubkey "$KEYS/payer.json")"

# The validator only deploys SBPF v3 programs. This uses the machine's cargo-build-sbf.
(cd program && cargo build-sbf --arch v3)

if ! curl -s -o /dev/null "$URL"; then
  mkdir -p "$LEDGER"
  # The payer is the genesis mint, so it holds the SOL that funds every run.
  nohup "$BIN/solana-test-validator" --ledger "$LEDGER" --reset --mint "$PAYER" --rpc-port "$PORT" --bind-address 127.0.0.1 --quiet >"$LEDGER/validator.log" 2>&1 &
  echo "validator starting (pid $!), ledger in $LEDGER"
fi
for _ in $(seq 60); do
  if [ "$(curl -s "$URL" -X POST -H 'Content-Type: application/json' -d '{"jsonrpc":"2.0","id":1,"method":"getHealth"}' | grep -c '"ok"')" = 1 ]; then break; fi
  sleep 1
done

"$BIN/solana" program deploy --url "$URL" --keypair "$KEYS/payer.json" --program-id "$KEYS/program.json" program/target/deploy/verillm.so
echo "program $("$BIN/solana-keygen" pubkey "$KEYS/program.json") deployed at $URL"
