MODEL_CFG=cfg/primitives/camera_ready/model_v3.yaml

# DATA_CFG=cfg/poke_experiments/rebuttal/mug_v3.yaml
# OUT_DIR=out/experiments/camera_ready/poke/mug_v3/
# python scripts/generate.py $MODEL_CFG -d $DATA_CFG -m test -o $OUT_DIR --gen_args "{"iter_limit": 100, "embed_weight": 0.001, "contact_threshold": 0.2}"

DATA_CFG=cfg/poke_experiments/rebuttal/bowl_v1.yaml
OUT_DIR=out/experiments/camera_ready/poke/bowl_v1/
python scripts/generate.py $MODEL_CFG -d $DATA_CFG -m test -o $OUT_DIR --gen_args "{"iter_limit": 100, "embed_weight": 0.001, "contact_threshold": 0.2}"
