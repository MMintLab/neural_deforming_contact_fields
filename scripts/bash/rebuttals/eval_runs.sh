#!/bin/bash

DATA_CFG=cfg/primitives/dataset/test_rebuttal.yaml
# MUG_DATA_CFG=cfg/poke_experiments/mug/mug_v6.yaml
MUG_DATA_CFG=cfg/
VAL_DATA_CFG=cfg/primitives/dataset/test_partial_v2.yaml

BASE_MODEL_CFG=cfg/primitives/model_v2.yaml

WRENCH_MODEL_CFG=cfg/primitives/rebuttal/wrench_v1.yaml
NO_WRENCH_MODEL_CFG=cfg/primitives/rebuttal/no_wrench_v1.yaml

OUT_DIR=out/experiments/rebuttal

#####################################
# Experiment 3: Poke Experiments    #
#####################################

# python scripts/generate.py $BASE_MODEL_CFG -d $MUG_DATA_CFG -m test -o $OUT_DIR/poke_experiments/mug_v6/v1 --gen_args "{"embed_weight": 0.0001, "def_loss": 0.0, "iter_limit": 50}"
# python scripts/generate.py $BASE_MODEL_CFG -d $MUG_DATA_CFG -m test -o $OUT_DIR/poke_experiments/mug_v6/v5 --gen_args "{"embed_weight": 0.001, "def_loss": 0.0, "iter_limit": 100}"
# python scripts/generate.py $BASE_MODEL_CFG -d $MUG_DATA_CFG -m test -o $OUT_DIR/poke_experiments/mug_v6/no_pc --gen_args "{"iter_limit": 0}"

################################################
# Experiment 1: Inference timing experiments   #
################################################

# Run 1: Original submission inference settings.
# python scripts/generate.py $BASE_MODEL_CFG -d $DATA_CFG -m test -o $OUT_DIR/inference_timing/submission/ --gen_args "{"embed_weight": 1.0, "def_loss": 0.0, "iter_limit": 1000}"
# python scripts/eval_results.py $DATA_CFG $OUT_DIR/inference_timing/submission/ -m test -s

# Run 2: New finetuned inference settings.
# python scripts/generate.py $BASE_MODEL_CFG -d $DATA_CFG -m test -o $OUT_DIR/inference_timing/finetuned/ --gen_args "{"embed_weight": 0.0001, "def_loss": 0.0, "iter_limit": 50}"
# python scripts/eval_results.py $DATA_CFG $OUT_DIR/inference_timing/finetuned/ -m test -s

# Run 3: No iterations.
# python scripts/generate.py $BASE_MODEL_CFG -d $DATA_CFG -m test -o $OUT_DIR/inference_timing/wrench_only/ --gen_args "{"iter_limit": 0}"
# python scripts/eval_results.py $DATA_CFG $OUT_DIR/inference_timing/wrench_only/ -m test -s

# Combine results.
# python scripts/plot/plot_inf_params_single.py $OUT_DIR/inference_timing/submission $OUT_DIR/inference_timing/finetuned $OUT_DIR/inference_timing/wrench_only $OUT_DIR/inference_timing/res.csv

####################################################
# Experiment 1: Full/Wrench-only/Pointcloud-only   #
####################################################

# Find best perf on validations set.
#python scripts/generate.py $WRENCH_MODEL_CFG -d $VAL_DATA_CFG -m test -o $OUT_DIR/input_ablations/val/full/0.001_25/ --gen_args "{"embed_weight": 0.001, "iter_limit": 25}"
#python scripts/generate.py $WRENCH_MODEL_CFG -d $VAL_DATA_CFG -m test -o $OUT_DIR/input_ablations/val/full/0.001_50/ --gen_args "{"embed_weight": 0.001, "iter_limit": 50}"
#python scripts/generate.py $WRENCH_MODEL_CFG -d $VAL_DATA_CFG -m test -o $OUT_DIR/input_ablations/val/full/0.001_100/ --gen_args "{"embed_weight": 0.001, "iter_limit": 100}"
#
#python scripts/generate.py $WRENCH_MODEL_CFG -d $VAL_DATA_CFG -m test -o $OUT_DIR/input_ablations/val/full/0.0001_25/ --gen_args "{"embed_weight": 0.0001, "iter_limit": 25}"
#python scripts/generate.py $WRENCH_MODEL_CFG -d $VAL_DATA_CFG -m test -o $OUT_DIR/input_ablations/val/full/0.0001_50/ --gen_args "{"embed_weight": 0.0001, "iter_limit": 50}"
#python scripts/generate.py $WRENCH_MODEL_CFG -d $VAL_DATA_CFG -m test -o $OUT_DIR/input_ablations/val/full/0.0001_100/ --gen_args "{"embed_weight": 0.0001, "iter_limit": 100}"
#
#python scripts/eval_results.py $VAL_DATA_CFG $OUT_DIR/input_ablations/val/full/0.001_25/ -m test -s
#python scripts/eval_results.py $VAL_DATA_CFG $OUT_DIR/input_ablations/val/full/0.001_50/ -m test -s
#python scripts/eval_results.py $VAL_DATA_CFG $OUT_DIR/input_ablations/val/full/0.001_100/ -m test -s
#
#python scripts/eval_results.py $VAL_DATA_CFG $OUT_DIR/input_ablations/val/full/0.0001_25/ -m test -s
#python scripts/eval_results.py $VAL_DATA_CFG $OUT_DIR/input_ablations/val/full/0.0001_50/ -m test -s
#python scripts/eval_results.py $VAL_DATA_CFG $OUT_DIR/input_ablations/val/full/0.0001_100/ -m test -s
#
#python scripts/plot/plot_inf_params_single.py $OUT_DIR/input_ablations/val/full/0.001_25/ $OUT_DIR/input_ablations/val/full/0.001_50/ $OUT_DIR/input_ablations/val/full/0.001_100/ $OUT_DIR/input_ablations/val/full/0.0001_25/ $OUT_DIR/input_ablations/val/full/0.0001_50/ $OUT_DIR/input_ablations/val/full/0.0001_100/ $OUT_DIR/input_ablations/val/val_res.csv

# Run 1: Full model.
# python scripts/generate.py $WRENCH_MODEL_CFG -d $DATA_CFG -m test -o $OUT_DIR/input_ablations/full/ --gen_args "{"embed_weight": 0.0001, "def_loss": 0.0, "iter_limit": 50}"
# python scripts/eval_results.py $DATA_CFG $OUT_DIR/input_ablations/full -m test -s

# Run 2: Wrench-only model.
# python scripts/generate.py $WRENCH_MODEL_CFG -d $DATA_CFG -m test -o $OUT_DIR/input_ablations/wrench_only/ --gen_args "{"embed_weight": 0.0001, "def_loss": 0.0, "iter_limit": 0}"
# python scripts/eval_results.py $DATA_CFG $OUT_DIR/input_ablations/wrench_only -m test -s

# Run 3: Pointcloud-only model.
# python scripts/generate.py $NO_WRENCH_MODEL_CFG -d $DATA_CFG -m test -o $OUT_DIR/input_ablations/pointcloud_only/ --gen_args "{"embed_weight": 0.0001, "def_loss": 0.0, "iter_limit": 50}"
# python scripts/eval_results.py $DATA_CFG $OUT_DIR/input_ablations/pointcloud_only -m test -s

# python scripts/plot/plot_inf_params_single.py $OUT_DIR/input_ablations/full $OUT_DIR/input_ablations/wrench_only $OUT_DIR/input_ablations/pointcloud_only $OUT_DIR/input_ablations/res.csv

####################################################
# Experiment 5: Iter count ablation                #
####################################################

#python scripts/generate.py $BASE_MODEL_CFG -d $DATA_CFG -m test -o $OUT_DIR/iter_ablation/1/ --gen_args "{"embed_weight": 0.0001, "def_loss": 0.0, "iter_limit": 1}"
#python scripts/generate.py $BASE_MODEL_CFG -d $DATA_CFG -m test -o $OUT_DIR/iter_ablation/50/ --gen_args "{"embed_weight": 0.0001, "def_loss": 0.0, "iter_limit": 50}"
#python scripts/generate.py $BASE_MODEL_CFG -d $DATA_CFG -m test -o $OUT_DIR/iter_ablation/100/ --gen_args "{"embed_weight": 0.0001, "def_loss": 0.0, "iter_limit": 100}"
#python scripts/generate.py $BASE_MODEL_CFG -d $DATA_CFG -m test -o $OUT_DIR/iter_ablation/200/ --gen_args "{"embed_weight": 0.0001, "def_loss": 0.0, "iter_limit": 200}"
#python scripts/generate.py $BASE_MODEL_CFG -d $DATA_CFG -m test -o $OUT_DIR/iter_ablation/300/ --gen_args "{"embed_weight": 0.0001, "def_loss": 0.0, "iter_limit": 300}"
#
#python scripts/eval_results.py $DATA_CFG $OUT_DIR/iter_ablation/1 -m test -s
#python scripts/eval_results.py $DATA_CFG $OUT_DIR/iter_ablation/50 -m test -s
#python scripts/eval_results.py $DATA_CFG $OUT_DIR/iter_ablation/100 -m test -s
#python scripts/eval_results.py $DATA_CFG $OUT_DIR/iter_ablation/200 -m test -s
#python scripts/eval_results.py $DATA_CFG $OUT_DIR/iter_ablation/300 -m test -s
#
#python scripts/plot/plot_inf_params_single.py $OUT_DIR/iter_ablation/1 $OUT_DIR/iter_ablation/50 $OUT_DIR/iter_ablation/100 $OUT_DIR/iter_ablation/200 $OUT_DIR/iter_ablation/300 $OUT_DIR/iter_ablation/res.csv
