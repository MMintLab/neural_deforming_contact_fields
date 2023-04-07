#!/bin/bash

DATA_CFG="cfg/primitives/dataset/test_partial_v2_noise.yaml"
OUT_BASE="out/experiments/primitives/model_v2/eval_inference_noise"

python scripts/eval_inference.py cfg/primitives/model_v2.yaml -d $DATA_CFG -m test -o $OUT_BASE/random/ --gen_args "{"iter_limit": 0}" -n 3

python scripts/eval_inference.py cfg/primitives/model_v2.yaml -d $DATA_CFG -m test -o $OUT_BASE/embed_weight/1.0/ --gen_args "{"embed_weight": 1.0}" -n 3
python scripts/eval_inference.py cfg/primitives/model_v2.yaml -d $DATA_CFG -m test -o $OUT_BASE/embed_weight/0.1/ --gen_args "{"embed_weight": 0.1}" -n 3
python scripts/eval_inference.py cfg/primitives/model_v2.yaml -d $DATA_CFG -m test -o $OUT_BASE/embed_weight/0.01/ --gen_args "{"embed_weight": 0.01}" -n 3
python scripts/eval_inference.py cfg/primitives/model_v2.yaml -d $DATA_CFG -m test -o $OUT_BASE/embed_weight/0.001/ --gen_args "{"embed_weight": 0.001}" -n 3
python scripts/eval_inference.py cfg/primitives/model_v2.yaml -d $DATA_CFG -m test -o $OUT_BASE/embed_weight/0.0001/ --gen_args "{"embed_weight": 0.0001}" -n 3
python scripts/eval_inference.py cfg/primitives/model_v2.yaml -d $DATA_CFG -m test -o $OUT_BASE/embed_weight/0/ --gen_args "{"embed_weight": 0.0}" -n 3

python scripts/eval_inference.py cfg/primitives/model_v2.yaml -d $DATA_CFG -m test -o $OUT_BASE/conv_eps/0.01/ --gen_args "{"conv_eps": 0.01}" -n 3
python scripts/eval_inference.py cfg/primitives/model_v2.yaml -d $DATA_CFG -m test -o $OUT_BASE/conv_eps/0.001/ --gen_args "{"conv_eps": 0.001}" -n 3
python scripts/eval_inference.py cfg/primitives/model_v2.yaml -d $DATA_CFG -m test -o $OUT_BASE/conv_eps/0.0001/ --gen_args "{"conv_eps": 0.0001}" -n 3
python scripts/eval_inference.py cfg/primitives/model_v2.yaml -d $DATA_CFG -m test -o $OUT_BASE/conv_eps/0.00001/ --gen_args "{"conv_eps": 0.00001}" -n 3
python scripts/eval_inference.py cfg/primitives/model_v2.yaml -d $DATA_CFG -m test -o $OUT_BASE/conv_eps/0.000001/ --gen_args "{"conv_eps": 0.000001}" -n 3

python scripts/eval_inference.py cfg/primitives/model_v2.yaml -d $DATA_CFG -m test -o $OUT_BASE/embed_conv/0.001_0.01/ --gen_args "{"embed_weight": 0.001, "conv_eps": 0.01}" -n 3
python scripts/eval_inference.py cfg/primitives/model_v2.yaml -d $DATA_CFG -m test -o $OUT_BASE/embed_conv/0.001_0.001/ --gen_args "{"embed_weight": 0.001, "conv_eps": 0.001}" -n 3
python scripts/eval_inference.py cfg/primitives/model_v2.yaml -d $DATA_CFG -m test -o $OUT_BASE/embed_conv/0.001_0.0001/ --gen_args "{"embed_weight": 0.001, "conv_eps": 0.0001}" -n 3
python scripts/eval_inference.py cfg/primitives/model_v2.yaml -d $DATA_CFG -m test -o $OUT_BASE/embed_conv/0.001_0.00001/ --gen_args "{"embed_weight": 0.001, "conv_eps": 0.00001}" -n 3
python scripts/eval_inference.py cfg/primitives/model_v2.yaml -d $DATA_CFG -m test -o $OUT_BASE/embed_conv/0.001_0.000001/ --gen_args "{"embed_weight": 0.001, "conv_eps": 0.000001}" -n 3

python scripts/eval_results.py $DATA_CFG $OUT_BASE/random/ -m test

python scripts/eval_results.py $DATA_CFG $OUT_BASE/embed_weight/1.0/ -m test
python scripts/eval_results.py $DATA_CFG $OUT_BASE/embed_weight/0.1/ -m test
python scripts/eval_results.py $DATA_CFG $OUT_BASE/embed_weight/0.01/ -m test
python scripts/eval_results.py $DATA_CFG $OUT_BASE/embed_weight/0.001/ -m test
python scripts/eval_results.py $DATA_CFG $OUT_BASE/embed_weight/0.0001/ -m test
python scripts/eval_results.py $DATA_CFG $OUT_BASE/embed_weight/0/ -m test

python scripts/eval_results.py $DATA_CFG $OUT_BASE/conv_eps/0.01/ -m test
python scripts/eval_results.py $DATA_CFG $OUT_BASE/conv_eps/0.001/ -m test
python scripts/eval_results.py $DATA_CFG $OUT_BASE/conv_eps/0.0001/ -m test
python scripts/eval_results.py $DATA_CFG $OUT_BASE/conv_eps/0.00001/ -m test
python scripts/eval_results.py $DATA_CFG $OUT_BASE/conv_eps/0.000001/ -m test

python scripts/eval_results.py $DATA_CFG $OUT_BASE/embed_conv/0.001_0.01/ -m test
python scripts/eval_results.py $DATA_CFG $OUT_BASE/embed_conv/0.001_0.001/ -m test
python scripts/eval_results.py $DATA_CFG $OUT_BASE/embed_conv/0.001_0.0001/ -m test
python scripts/eval_results.py $DATA_CFG $OUT_BASE/embed_conv/0.001_0.00001/ -m test
python scripts/eval_results.py $DATA_CFG $OUT_BASE/embed_conv/0.001_0.000001/ -m test

python scripts/plot/plot_inf_params.py $OUT_BASE/random/ $OUT_BASE/embed_weight/1.0/ $OUT_BASE/embed_weight/0.1/ $OUT_BASE/embed_weight/0.01/ $OUT_BASE/embed_weight/0.001/ $OUT_BASE/embed_weight/0.0001/ $OUT_BASE/embed_weight/0/ $OUT_BASE/embed_weight.csv
python scripts/plot/plot_inf_params.py $OUT_BASE/random/ $OUT_BASE/conv_eps/0.01/ $OUT_BASE/conv_eps/0.001/ $OUT_BASE/conv_eps/0.0001/ $OUT_BASE/conv_eps/0.00001/ $OUT_BASE/conv_eps/0.000001/ $OUT_BASE/conv_eps.csv
python scripts/plot/plot_inf_params.py $OUT_BASE/random/ $OUT_BASE/embed_conv/0.001_0.01/ $OUT_BASE/embed_conv/0.001_0.001/ $OUT_BASE/embed_conv/0.001_0.0001/ $OUT_BASE/embed_conv/0.001_0.00001/ $OUT_BASE/embed_conv/0.001_0.000001/ $OUT_BASE/embed_conv.csv
