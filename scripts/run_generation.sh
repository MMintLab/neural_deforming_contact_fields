#python scripts/generate.py cfg/wrench_tests/wrench_v1.yaml -d cfg/dataset/pretrain_tests_gen.yaml -m test -o out/experiments/wrench_tests/wrench_v1/
#python scripts/generate.py cfg/wrench_tests/wrench_v2.yaml -d cfg/dataset/pretrain_tests_gen.yaml -m test -o out/experiments/wrench_tests/wrench_v2/
#python scripts/generate.py cfg/wrench_tests/wrench_v3.yaml -d cfg/dataset/pretrain_tests_gen.yaml -m test -o out/experiments/wrench_tests/wrench_v3/
#python scripts/generate.py cfg/wrench_tests/wrench_v4.yaml -d cfg/dataset/pretrain_tests_gen.yaml -m test -o out/experiments/wrench_tests/wrench_v4/
#python scripts/generate.py cfg/wrench_tests/wrench_v5.yaml -d cfg/dataset/pretrain_tests_gen.yaml -m test -o out/experiments/wrench_tests/wrench_v5/
#python scripts/generate.py cfg/wrench_tests/wrench_v6.yaml -d cfg/dataset/pretrain_tests_gen.yaml -m test -o out/experiments/wrench_tests/wrench_v6/
#python scripts/generate.py cfg/wrench_tests/wrench_v7.yaml -d cfg/dataset/pretrain_tests_gen.yaml -m test -o out/experiments/wrench_tests/wrench_v7/
#python scripts/generate.py cfg/wrench_tests/wrench_v8.yaml -d cfg/dataset/pretrain_tests_gen.yaml -m test -o out/experiments/wrench_tests/wrench_v8/
#python scripts/generate.py cfg/wrench_tests/wrench_v9.yaml -d cfg/dataset/pretrain_tests_gen.yaml -m test -o out/experiments/wrench_tests/wrench_v9/
#python scripts/generate.py cfg/wrench_tests/wrench_v10.yaml -d cfg/dataset/pretrain_tests_gen.yaml -m test -o out/experiments/wrench_tests/wrench_v10/
#python scripts/generate.py cfg/wrench_tests/mlp_v1.yaml -d cfg/dataset/pretrain_tests_gen.yaml -m test -o out/experiments/wrench_tests/mlp_v1/
#python scripts/generate.py cfg/wrench_tests/no_wrench_v1.yaml -d cfg/dataset/pretrain_tests_gen.yaml -m test -o out/experiments/wrench_tests/no_wrench_v1/

python scripts/eval_results.py cfg/dataset/pretrain_tests_gen.yaml out/experiments/wrench_tests/wrench_v1 -m test
python scripts/eval_results.py cfg/dataset/pretrain_tests_gen.yaml out/experiments/wrench_tests/wrench_v2 -m test
python scripts/eval_results.py cfg/dataset/pretrain_tests_gen.yaml out/experiments/wrench_tests/wrench_v3 -m test
python scripts/eval_results.py cfg/dataset/pretrain_tests_gen.yaml out/experiments/wrench_tests/wrench_v4 -m test
python scripts/eval_results.py cfg/dataset/pretrain_tests_gen.yaml out/experiments/wrench_tests/wrench_v5 -m test
python scripts/eval_results.py cfg/dataset/pretrain_tests_gen.yaml out/experiments/wrench_tests/wrench_v6 -m test
python scripts/eval_results.py cfg/dataset/pretrain_tests_gen.yaml out/experiments/wrench_tests/wrench_v7 -m test
python scripts/eval_results.py cfg/dataset/pretrain_tests_gen.yaml out/experiments/wrench_tests/wrench_v8 -m test
python scripts/eval_results.py cfg/dataset/pretrain_tests_gen.yaml out/experiments/wrench_tests/wrench_v9 -m test
python scripts/eval_results.py cfg/dataset/pretrain_tests_gen.yaml out/experiments/wrench_tests/wrench_v10 -m test
python scripts/eval_results.py cfg/dataset/pretrain_tests_gen.yaml out/experiments/wrench_tests/no_wrench_v1/ -m test
python scripts/eval_results.py cfg/dataset/pretrain_tests_gen.yaml out/experiments/wrench_tests/mlp_v1 -m test

