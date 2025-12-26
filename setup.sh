curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
. "$HOME/.cargo/env"
apt install build-essential -y

export RUSTFLAGS='-C target-cpu=native'
export RUSTUP_TOOLCHAIN=nightly