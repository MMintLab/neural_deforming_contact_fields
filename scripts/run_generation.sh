# python scripts/generate.py cfg/wrench_v2_tests/wrench_v1.yaml -d cfg/dataset/wrench_v2_tests.yaml -m test -o out/experiments/wrench_v2_tests/partial_pointcloud/wrench_v1/
python scripts/generate.py cfg/wrench_v2_tests/wrench_v2.yaml -d cfg/dataset/wrench_v2_tests.yaml -m test -o out/experiments/wrench_v2_tests/partial_pointcloud/wrench_v2/
python scripts/generate.py cfg/wrench_v2_tests/wrench_v3.yaml -d cfg/dataset/wrench_v2_tests.yaml -m test -o out/experiments/wrench_v2_tests/partial_pointcloud/wrench_v3/
python scripts/generate.py cfg/wrench_v2_tests/wrench_v4.yaml -d cfg/dataset/wrench_v2_tests.yaml -m test -o out/experiments/wrench_v2_tests/partial_pointcloud/wrench_v4/
python scripts/generate.py cfg/wrench_v2_tests/no_wrench_v1.yaml -d cfg/dataset/wrench_v2_tests.yaml -m test -o out/experiments/wrench_v2_tests/partial_pointcloud/no_wrench_v1/
python scripts/generate.py cfg/wrench_v2_tests/no_wrench_v2.yaml -d cfg/dataset/wrench_v2_tests.yaml -m test -o out/experiments/wrench_v2_tests/partial_pointcloud/no_wrench_v2/

python scripts/eval_results.py cfg/dataset/wrench_v2_tests.yaml out/experiments/wrench_v2_tests/partial_pointcloud/wrench_v1 -m test
python scripts/eval_results.py cfg/dataset/wrench_v2_tests.yaml out/experiments/wrench_v2_tests/partial_pointcloud/wrench_v2 -m test
python scripts/eval_results.py cfg/dataset/wrench_v2_tests.yaml out/experiments/wrench_v2_tests/partial_pointcloud/wrench_v3 -m test
python scripts/eval_results.py cfg/dataset/wrench_v2_tests.yaml out/experiments/wrench_v2_tests/partial_pointcloud/wrench_v4 -m test
python scripts/eval_results.py cfg/dataset/wrench_v2_tests.yaml out/experiments/wrench_v2_tests/partial_pointcloud/no_wrench_v1 -m test
python scripts/eval_results.py cfg/dataset/wrench_v2_tests.yaml out/experiments/wrench_v2_tests/partial_pointcloud/no_wrench_v2 -m test
