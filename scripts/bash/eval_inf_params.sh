#python scripts/eval_inference.py cfg/primitives/model_v2.yaml -d cfg/primitives/dataset/test_partial_v2.yaml -m test -o out/experiments/primitives/model_v2/eval_inference/random/ --gen_args "{"iter_limit": 0}" -n 3
#
#python scripts/eval_inference.py cfg/primitives/model_v2.yaml -d cfg/primitives/dataset/test_partial_v2.yaml -m test -o out/experiments/primitives/model_v2/eval_inference/embed_weight/1.0/ --gen_args "{"embed_weight": 1.0}" -n 3
#python scripts/eval_inference.py cfg/primitives/model_v2.yaml -d cfg/primitives/dataset/test_partial_v2.yaml -m test -o out/experiments/primitives/model_v2/eval_inference/embed_weight/0.1/ --gen_args "{"embed_weight": 0.1}" -n 3
#python scripts/eval_inference.py cfg/primitives/model_v2.yaml -d cfg/primitives/dataset/test_partial_v2.yaml -m test -o out/experiments/primitives/model_v2/eval_inference/embed_weight/0.01/ --gen_args "{"embed_weight": 0.01}" -n 3
#python scripts/eval_inference.py cfg/primitives/model_v2.yaml -d cfg/primitives/dataset/test_partial_v2.yaml -m test -o out/experiments/primitives/model_v2/eval_inference/embed_weight/0.001/ --gen_args "{"embed_weight": 0.001}" -n 3
#python scripts/eval_inference.py cfg/primitives/model_v2.yaml -d cfg/primitives/dataset/test_partial_v2.yaml -m test -o out/experiments/primitives/model_v2/eval_inference/embed_weight/0.0001/ --gen_args "{"embed_weight": 0.0001}" -n 3
#
python scripts/eval_inference.py cfg/primitives/model_v2.yaml -d cfg/primitives/dataset/test_partial_v2.yaml -m test -o out/experiments/primitives/model_v2/eval_inference/conv_eps/0.01/ --gen_args "{"conv_eps": 0.01}" -n 3
python scripts/eval_inference.py cfg/primitives/model_v2.yaml -d cfg/primitives/dataset/test_partial_v2.yaml -m test -o out/experiments/primitives/model_v2/eval_inference/conv_eps/0.001/ --gen_args "{"conv_eps": 0.001}" -n 3
python scripts/eval_inference.py cfg/primitives/model_v2.yaml -d cfg/primitives/dataset/test_partial_v2.yaml -m test -o out/experiments/primitives/model_v2/eval_inference/conv_eps/0.0001/ --gen_args "{"conv_eps": 0.0001}" -n 3
python scripts/eval_inference.py cfg/primitives/model_v2.yaml -d cfg/primitives/dataset/test_partial_v2.yaml -m test -o out/experiments/primitives/model_v2/eval_inference/conv_eps/0.00001/ --gen_args "{"conv_eps": 0.00001}" -n 3
python scripts/eval_inference.py cfg/primitives/model_v2.yaml -d cfg/primitives/dataset/test_partial_v2.yaml -m test -o out/experiments/primitives/model_v2/eval_inference/conv_eps/0.000001/ --gen_args "{"conv_eps": 0.000001}" -n 3

python scripts/eval_inference.py cfg/primitives/model_v2.yaml -d cfg/primitives/dataset/test_partial_v2.yaml -m test -o out/experiments/primitives/model_v2/eval_inference/embed_conv/0.001_0.01/ --gen_args "{"embed_weight": 0.001, "conv_eps": 0.01}" -n 3
python scripts/eval_inference.py cfg/primitives/model_v2.yaml -d cfg/primitives/dataset/test_partial_v2.yaml -m test -o out/experiments/primitives/model_v2/eval_inference/embed_conv/0.001_0.001/ --gen_args "{"embed_weight": 0.001, "conv_eps": 0.001}" -n 3
python scripts/eval_inference.py cfg/primitives/model_v2.yaml -d cfg/primitives/dataset/test_partial_v2.yaml -m test -o out/experiments/primitives/model_v2/eval_inference/embed_conv/0.001_0.0001/ --gen_args "{"embed_weight": 0.001, "conv_eps": 0.0001}" -n 3
