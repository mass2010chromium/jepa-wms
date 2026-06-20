SCRIPT_DIR=$(cd -- "$(dirname -- "$BASH_SOURCE[0]")/" && pwd)

cd "$SCRIPT_DIR"
mkdir -p logs
mkdir -p checkpoints

source set_env.sh
# hf download hf://facebook/jepa-wms/dinov2_vits_224_INet.pth.tar --local-dir checkpoints
hf download hf://facebook/jepa-wms/jepa_wm_pusht.pth.tar --local-dir checkpoints

cd "$SCRIPT_DIR/logs/pt_sweep/pt_4f_fsk5_ask1_r224_vjtranoaug_predAdaLN_ftprop_depth6_repro_2roll_save/"
ln -sT ../../../jepa_wm_pusht.pth.tar jepa-latest.pth.tar
