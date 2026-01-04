apt install build-essential -y

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
. "$HOME/.cargo/env"

export RUSTFLAGS='-C target-cpu=native'
export RUSTUP_TOOLCHAIN=nightly