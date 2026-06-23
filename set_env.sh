SCRIPT_DIR=$(cd -- "$(dirname -- "$BASH_SOURCE[0]")/" && pwd)

export JEPAWM_DSET="$SCRIPT_DIR/datasets"
export JEPAWM_LOGS="$SCRIPT_DIR/logs"
export JEPAWM_HOME="$SCRIPT_DIR/.."
export JEPAWM_CKPT="$SCRIPT_DIR/checkpoints"
export JEPAWM_OSSCKPT="$SCRIPT_DIR/oss_checkpoints"
#source "$SCRIPT_DIR/.venv/bin/activate"
