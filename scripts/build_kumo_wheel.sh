#!/usr/bin/env bash
# Build the kumo-engine wheel that requirements.txt pins, into scratch/wheels/.
#
# PyPI's kumo-engine 1.0.1 has only a Linux x86_64 wheel, so requirements.txt pins a
# local version, 1.0.0.dev0+<commit>, built here from NVIDIA/kumo-engine at that commit.
# `make install` calls this. Re-running it does nothing once the wheel exists, and it
# does nothing at all when requirements.txt pins a plain PyPI version.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT=$PWD
PYTHON=${PYTHON:-python3}
SRC=scratch/kumo-src
BUILD_VENV=scratch/build-venv
WHEELS=scratch/wheels
LOG=scratch/build.log

say() { printf '==> %s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

"$PYTHON" -c 'import sys; sys.exit(sys.version_info < (3, 11))' 2>/dev/null \
  || die "kumo-engine needs Python 3.11 or newer, and '$PYTHON' isn't; try: make install PYTHON=python3.14"

VERSION=$(sed -n 's/^kumo-engine==//p' requirements.txt)
[[ -n $VERSION ]] || die "requirements.txt has no 'kumo-engine==' line"
if [[ $VERSION != *+* ]]; then
  say "requirements.txt pins kumo-engine $VERSION from PyPI; no wheel to build"
  exit 0
fi
COMMIT=${VERSION#*+}
if compgen -G "$WHEELS/kumo_engine-$VERSION-*.whl" >/dev/null; then
  say "kumo-engine $VERSION is already built in $WHEELS/"
  exit 0
fi

# Rust. Homebrew's rustup is keg-only, so its cargo and rustc aren't on PATH by default.
export PATH="$HOME/.cargo/bin:/opt/homebrew/opt/rustup/bin:/usr/local/opt/rustup/bin:$PATH"
if ! command -v rustup >/dev/null; then
  command -v brew >/dev/null \
    || die "building kumo-engine needs Rust: install rustup from https://rustup.rs, then re-run make install"
  say "Installing rustup with Homebrew"
  brew install rustup
fi

say "Fetching NVIDIA/kumo-engine at $COMMIT into $SRC/"
[[ -d $SRC/.git ]] || git clone --quiet --filter=blob:none https://github.com/NVIDIA/kumo-engine.git "$SRC"
git -C "$SRC" cat-file -e "$COMMIT^{commit}" 2>/dev/null || git -C "$SRC" fetch --quiet origin
git -C "$SRC" checkout --quiet --detach "$COMMIT"

TOOLCHAIN=$(sed -n 's/^channel *= *"\(.*\)"/\1/p' "$SRC/packages/diskgraph/rust-toolchain.toml")
say "Installing Rust $TOOLCHAIN, the toolchain kumo-engine pins"
rustup toolchain install "$TOOLCHAIN" --profile minimal

say "Preparing the build environment in $BUILD_VENV/"
[[ -x $BUILD_VENV/bin/python ]] || "$PYTHON" -m venv "$BUILD_VENV"
"$BUILD_VENV/bin/python" -m pip install --quiet --upgrade pip wheel

say "Building the wheel; compiling DiskGraph takes a few minutes (log: $LOG)"
# BUNDLE_VERSION is set because the build otherwise stamps `git rev-parse --short HEAD`,
# which can be longer than the hash requirements.txt pins (a fresh clone gives 5d17fb3de).
if ! (cd "$SRC" && RUSTUP_TOOLCHAIN="$TOOLCHAIN" PYTHON="$ROOT/$BUILD_VENV/bin/python" \
        OUTPUT_DIR="$ROOT/$WHEELS" BUNDLE_VERSION="$VERSION" \
        ./packages/kumo-engine/scripts/build_wheel.sh) >"$LOG" 2>&1; then
  tail -n 40 "$LOG" >&2
  die "the wheel build failed; the full log is in $LOG"
fi
say "Built $(ls "$WHEELS"/kumo_engine-"$VERSION"-*.whl)"
