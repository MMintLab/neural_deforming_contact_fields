python scripts/generate.py cfg/terrain_tests/wrench_v1.yaml -d cfg/dataset/sim_terrain_test.yaml -m test -o out/experiments/terrain_tests/partial_pointcloud/wrench_v1/
python scripts/generate.py cfg/terrain_tests/wrench_v2.yaml -d cfg/dataset/sim_terrain_test.yaml -m test -o out/experiments/terrain_tests/partial_pointcloud/wrench_v2/
python scripts/generate.py cfg/terrain_tests/wrench_v3.yaml -d cfg/dataset/sim_terrain_test.yaml -m test -o out/experiments/terrain_tests/partial_pointcloud/wrench_v3/
python scripts/generate.py cfg/terrain_tests/wrench_v4.yaml -d cfg/dataset/sim_terrain_test.yaml -m test -o out/experiments/terrain_tests/partial_pointcloud/wrench_v4/

python scripts/eval_results.py cfg/dataset/sim_terrain_test.yaml out/experiments/terrain_tests/partial_pointcloud/wrench_v1 -m test
python scripts/eval_results.py cfg/dataset/sim_terrain_test.yaml out/experiments/terrain_tests/partial_pointcloud/wrench_v2 -m test
python scripts/eval_results.py cfg/dataset/sim_terrain_test.yaml out/experiments/terrain_tests/partial_pointcloud/wrench_v3 -m test
python scripts/eval_results.py cfg/dataset/sim_terrain_test.yaml out/experiments/terrain_tests/partial_pointcloud/wrench_v4 -m test
