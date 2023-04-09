#!/bin/bash

MUG_CFG=cfg/poke_experiments/rebuttal/mug_v1.yaml

MODEL_CFG=cfg/primitives/model_v2.yaml

OUT_DIR=out/rebuttal/poke_experiments/updated/

python scripts/generate.py $MODEL_CFG -d $MUG_CFG -m test -o $OUT_DIR/mug_v1/ --gen_args "{"embed_weight": 0.001, "def_loss": 0.0, "iter_limit": 100}"
