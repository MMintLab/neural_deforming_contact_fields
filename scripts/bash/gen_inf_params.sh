#!/bin/bash

MODEL_CFG="cfg/primitives/model_v2.yaml"
DATA_CFG="cfg/primitives/dataset/test_partial_v2.yaml"
OUT_BASE="out/experiments/primitives/model_v2/eval_inference_gen"

#python scripts/eval_inference.py $MODEL_CFG -d $DATA_CFG -m test -o $OUT_BASE/random/ --gen_args "{"iter_limit": 0}" -n 3
#
#python scripts/eval_inference.py $MODEL_CFG -d $DATA_CFG -m test -o $OUT_BASE/embed_weight/1.0/ --gen_args "{"embed_weight": 1.0}" -n 3
#python scripts/eval_inference.py $MODEL_CFG -d $DATA_CFG -m test -o $OUT_BASE/embed_weight/0.1/ --gen_args "{"embed_weight": 0.1}" -n 3
#python scripts/eval_inference.py $MODEL_CFG -d $DATA_CFG -m test -o $OUT_BASE/embed_weight/0.01/ --gen_args "{"embed_weight": 0.01}" -n 3
#python scripts/eval_inference.py $MODEL_CFG -d $DATA_CFG -m test -o $OUT_BASE/embed_weight/0.001/ --gen_args "{"embed_weight": 0.001}" -n 3
#python scripts/eval_inference.py $MODEL_CFG -d $DATA_CFG -m test -o $OUT_BASE/embed_weight/0.0001/ --gen_args "{"embed_weight": 0.0001}" -n 3
# python scripts/eval_inference.py $MODEL_CFG -d $DATA_CFG -m test -o $OUT_BASE/embed_weight/0/ --gen_args "{"embed_weight": 0.0}" -n 3
#
#python scripts/eval_inference.py $MODEL_CFG -d $DATA_CFG -m test -o $OUT_BASE/conv_eps/0.01/ --gen_args "{"conv_eps": 0.01}" -n 3
#python scripts/eval_inference.py $MODEL_CFG -d $DATA_CFG -m test -o $OUT_BASE/conv_eps/0.001/ --gen_args "{"conv_eps": 0.001}" -n 3
#python scripts/eval_inference.py $MODEL_CFG -d $DATA_CFG -m test -o $OUT_BASE/conv_eps/0.0001/ --gen_args "{"conv_eps": 0.0001}" -n 3
#python scripts/eval_inference.py $MODEL_CFG -d $DATA_CFG -m test -o $OUT_BASE/conv_eps/0.00001/ --gen_args "{"conv_eps": 0.00001}" -n 3
#python scripts/eval_inference.py $MODEL_CFG -d $DATA_CFG -m test -o $OUT_BASE/conv_eps/0.000001/ --gen_args "{"conv_eps": 0.000001}" -n 3
#
#python scripts/eval_inference.py $MODEL_CFG -d $DATA_CFG -m test -o $OUT_BASE/embed_conv/0.001_0.01/ --gen_args "{"embed_weight": 0.001, "conv_eps": 0.01}" -n 3
#python scripts/eval_inference.py $MODEL_CFG -d $DATA_CFG -m test -o $OUT_BASE/embed_conv/0.001_0.001/ --gen_args "{"embed_weight": 0.001, "conv_eps": 0.001}" -n 3
#python scripts/eval_inference.py $MODEL_CFG -d $DATA_CFG -m test -o $OUT_BASE/embed_conv/0.001_0.0001/ --gen_args "{"embed_weight": 0.001, "conv_eps": 0.0001}" -n 3
#python scripts/eval_inference.py $MODEL_CFG -d $DATA_CFG -m test -o $OUT_BASE/embed_conv/0.001_0.00001/ --gen_args "{"embed_weight": 0.001, "conv_eps": 0.00001}" -n 3
#python scripts/eval_inference.py $MODEL_CFG -d $DATA_CFG -m test -o $OUT_BASE/embed_conv/0.001_0.000001/ --gen_args "{"embed_weight": 0.001, "conv_eps": 0.000001}" -n 3

#python scripts/eval_inference.py $MODEL_CFG -d $DATA_CFG -m test -o $OUT_BASE/iter_limit/50/ --gen_args "{"iter_limit": 50}" -n 3
#python scripts/eval_inference.py $MODEL_CFG -d $DATA_CFG -m test -o $OUT_BASE/iter_limit/60/ --gen_args "{"iter_limit": 60}" -n 3
#python scripts/eval_inference.py $MODEL_CFG -d $DATA_CFG -m test -o $OUT_BASE/iter_limit/100/ --gen_args "{"iter_limit": 100}" -n 3
#python scripts/eval_inference.py $MODEL_CFG -d $DATA_CFG -m test -o $OUT_BASE/iter_limit/300/ --gen_args "{"iter_limit": 300}" -n 3

# python scripts/generate.py $MODEL_CFG -d $DATA_CFG -m test -o $OUT_BASE/random/ --gen_args "{"iter_limit": 0}"
#
# python scripts/generate.py $MODEL_CFG -d $DATA_CFG -m test -o $OUT_BASE/embed_weight_std/0.0001/ --gen_args "{"embed_weight": 0.0001}"
#python scripts/generate.py $MODEL_CFG -d $DATA_CFG -m test -o $OUT_BASE/embed_weight/0/ --gen_args "{"embed_weight": 0.0}"
# python scripts/generate.py $MODEL_CFG -d $DATA_CFG -m test -o $OUT_BASE/embed_weight/0_1000/ --gen_args "{"embed_weight": 0.0, "iter_limit": 1000}"

#python scripts/generate.py $MODEL_CFG -d $DATA_CFG -m test -o $OUT_BASE/embed_iter/0.001_50/ --gen_args "{"embed_weight": 0.001, "iter_limit": 50}"
#python scripts/generate.py $MODEL_CFG -d $DATA_CFG -m test -o $OUT_BASE/embed_iter/0.001_100/ --gen_args "{"embed_weight": 0.001, "iter_limit": 100}"
#python scripts/generate.py $MODEL_CFG -d $DATA_CFG -m test -o $OUT_BASE/embed_iter/0.001_150/ --gen_args "{"embed_weight": 0.001, "iter_limit": 150}"
#
#python scripts/generate.py $MODEL_CFG -d $DATA_CFG -m test -o $OUT_BASE/embed_iter/0.0001_50/ --gen_args "{"embed_weight": 0.0001, "iter_limit": 50}"
#python scripts/generate.py $MODEL_CFG -d $DATA_CFG -m test -o $OUT_BASE/embed_iter/0.0001_100/ --gen_args "{"embed_weight": 0.0001, "iter_limit": 100}"
# python scripts/generate.py $MODEL_CFG -d $DATA_CFG -m test -o $OUT_BASE/embed_iter/0.0001_150/ --gen_args "{"embed_weight": 0.0001, "iter_limit": 150}"

python scripts/generate.py $MODEL_CFG -d $DATA_CFG -m test -o $OUT_BASE/embed_iter/1.0_1000/ --gen_args "{"embed_weight": 1.0, "iter_limit": 1000}"

#python scripts/generate.py $MODEL_CFG -d $DATA_CFG -m test -o $OUT_BASE/embed_conv/0.001_0.0001/ --gen_args "{"embed_weight": 0.001, "conv_eps": 0.0001}"

# python scripts/eval_results.py $DATA_CFG $OUT_BASE/random/ -m test
#
#python scripts/eval_results.py $DATA_CFG $OUT_BASE/embed_weight/1.0/ -m test
#python scripts/eval_results.py $DATA_CFG $OUT_BASE/embed_weight/0.1/ -m test
#python scripts/eval_results.py $DATA_CFG $OUT_BASE/embed_weight/0.01/ -m test
#python scripts/eval_results.py $DATA_CFG $OUT_BASE/embed_weight/0.001/ -m test
#python scripts/eval_results.py $DATA_CFG $OUT_BASE/embed_weight/0.0001/ -m test
#python scripts/eval_results.py $DATA_CFG $OUT_BASE/embed_weight/0/ -m test
#
#python scripts/eval_results.py $DATA_CFG $OUT_BASE/conv_eps/0.01/ -m test
#python scripts/eval_results.py $DATA_CFG $OUT_BASE/conv_eps/0.001/ -m test
#python scripts/eval_results.py $DATA_CFG $OUT_BASE/conv_eps/0.0001/ -m test
#python scripts/eval_results.py $DATA_CFG $OUT_BASE/conv_eps/0.00001/ -m test
#python scripts/eval_results.py $DATA_CFG $OUT_BASE/conv_eps/0.000001/ -m test
#
#python scripts/eval_results.py $DATA_CFG $OUT_BASE/embed_conv/0.001_0.01/ -m test
#python scripts/eval_results.py $DATA_CFG $OUT_BASE/embed_conv/0.001_0.001/ -m test
#python scripts/eval_results.py $DATA_CFG $OUT_BASE/embed_conv/0.001_0.0001/ -m test
#python scripts/eval_results.py $DATA_CFG $OUT_BASE/embed_conv/0.001_0.00001/ -m test
#python scripts/eval_results.py $DATA_CFG $OUT_BASE/embed_conv/0.001_0.000001/ -m test

#python scripts/eval_results.py $DATA_CFG $OUT_BASE/iter_limit/50/ -m test
#python scripts/eval_results.py $DATA_CFG $OUT_BASE/iter_limit/60/ -m test
#python scripts/eval_results.py $DATA_CFG $OUT_BASE/iter_limit/100/ -m test
#python scripts/eval_results.py $DATA_CFG $OUT_BASE/iter_limit/300/ -m test

#python scripts/eval_results.py $DATA_CFG $OUT_BASE/embed_iter/0.001_50/ -m test
#python scripts/eval_results.py $DATA_CFG $OUT_BASE/embed_iter/0.001_100/ -m test
#python scripts/eval_results.py $DATA_CFG $OUT_BASE/embed_iter/0.001_150/ -m test
#
#python scripts/eval_results.py $DATA_CFG $OUT_BASE/embed_iter/0.0001_50/ -m test
#python scripts/eval_results.py $DATA_CFG $OUT_BASE/embed_iter/0.0001_100/ -m test
#python scripts/eval_results.py $DATA_CFG $OUT_BASE/embed_iter/0.0001_150/ -m test

python scripts/eval_results.py $DATA_CFG $OUT_BASE/embed_iter/1.0_1000/ -m test

#python scripts/eval_results.py $DATA_CFG $OUT_BASE/random/ -m test

# python scripts/eval_results.py $DATA_CFG $OUT_BASE/embed_weight_std/0.0001/ -m test

#python scripts/eval_results.py $DATA_CFG $OUT_BASE/embed_weight/0.0001/ -m test
#python scripts/eval_results.py $DATA_CFG $OUT_BASE/embed_weight/0/ -m test
#python scripts/eval_results.py $DATA_CFG $OUT_BASE/embed_weight/0_1000/ -m test
#
#python scripts/eval_results.py $DATA_CFG $OUT_BASE/embed_conv/0.001_0.0001/ -m test

#python scripts/plot/plot_inf_params.py $OUT_BASE/random/ $OUT_BASE/embed_weight/1.0/ $OUT_BASE/embed_weight/0.1/ $OUT_BASE/embed_weight/0.01/ $OUT_BASE/embed_weight/0.001/ $OUT_BASE/embed_weight/0.0001/ $OUT_BASE/embed_weight/0/ $OUT_BASE/embed_weight.csv
#python scripts/plot/plot_inf_params.py $OUT_BASE/random/ $OUT_BASE/conv_eps/0.01/ $OUT_BASE/conv_eps/0.001/ $OUT_BASE/conv_eps/0.0001/ $OUT_BASE/conv_eps/0.00001/ $OUT_BASE/conv_eps/0.000001/ $OUT_BASE/conv_eps.csv
#python scripts/plot/plot_inf_params.py $OUT_BASE/random/ $OUT_BASE/embed_conv/0.001_0.01/ $OUT_BASE/embed_conv/0.001_0.001/ $OUT_BASE/embed_conv/0.001_0.0001/ $OUT_BASE/embed_conv/0.001_0.00001/ $OUT_BASE/embed_conv/0.001_0.000001/ $OUT_BASE/embed_conv.csv
#python scripts/plot/plot_inf_params.py $OUT_BASE/random/ $OUT_BASE/iter_limit/50/ $OUT_BASE/iter_limit/60/ $OUT_BASE/iter_limit/100/ $OUT_BASE/iter_limit/300/ $OUT_BASE/iter_limit.csv

# python scripts/plot/plot_inf_params_single.py $OUT_BASE/random/ $OUT_BASE/embed_weight/0.0001/ $OUT_BASE/embed_weight/0/ $OUT_BASE/embed_weight/0_1000/ $OUT_BASE/embed_conv/0.001_0.0001/ $OUT_BASE/embed_weight_std/std_0.0001/ $OUT_BASE/gen_res.csv

python scripts/plot/plot_inf_params_single.py $OUT_BASE/random/ $OUT_BASE/embed_iter/0.001_50/ $OUT_BASE/embed_iter/0.001_100/ $OUT_BASE/embed_iter/0.001_150/ $OUT_BASE/embed_iter/0.0001_50/ $OUT_BASE/embed_iter/0.0001_100/ $OUT_BASE/embed_iter/0.0001_150/ $OUT_BASE/embed_iter/1.0_1000/ $OUT_BASE/embed_iter.csv
