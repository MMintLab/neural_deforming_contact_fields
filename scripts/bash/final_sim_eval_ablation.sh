#python scripts/generate.py cfg/terrain_tests/wrench_v2.yaml -d cfg/dataset/sim_terrain_test_v2.yaml -m test -o out/experiments/terrain_tests_v2/partial_pointcloud/wrench_v2/ --gen_args "{"embed_weight": 1.0, "def_loss": 0.0}"
#python scripts/eval_results.py cfg/dataset/sim_terrain_test_v2.yaml out/experiments/terrain_tests_v2/partial_pointcloud/wrench_v2 -m test

#python scripts/eval_results.py cfg/dataset/sim_terrain_test_v2.yaml out/experiments/terrain_tests_v2/partial_pointcloud/baseline -m test

# python scripts/generate.py cfg/terrain_tests/wrench_v2.yaml -d cfg/dataset/sim_terrain_test_v2.yaml -m test -o out/experiments/terrain_tests_v2/partial_pointcloud_vis/wrench_v2/ --gen_args "{"embed_weight": 1.0, "def_loss": 0.0}"

#python scripts/generate.py cfg/terrain_tests/wrench_v2.yaml -d cfg/dataset/sim_terrain_test_v2.yaml -m test -o out/experiments/terrain_tests_v3/partial_pointcloud/wrench_v2/ --gen_args "{"embed_weight": 1.0, "def_loss": 0.0}"

#python scripts/eval_results.py cfg/dataset/sim_terrain_test_v2.yaml out/experiments/terrain_tests_v3/partial_pointcloud/wrench_v2 -m test -s
#python scripts/eval_results.py cfg/dataset/sim_terrain_test_v2.yaml out/experiments/terrain_tests_v3/partial_pointcloud/baseline -m test -s

python scripts/generate.py cfg/terrain_tests/no_wrench_v1.yaml -d cfg/dataset/sim_terrain_test_v2.yaml -m test -o out/experiments/terrain_tests_v3/partial_pointcloud/no_wrench_v1/ --gen_args "{"embed_weight": 1.0, "def_loss": 0.0}"
python scripts/generate.py cfg/terrain_tests/forward_def_v1.yaml -d cfg/dataset/sim_terrain_test_v2.yaml -m test -o out/experiments/terrain_tests_v3/partial_pointcloud/forward_def_v1/ --gen_args "{"embed_weight": 1.0, "def_loss": 0.0}"

python scripts/eval_results.py cfg/dataset/sim_terrain_test_v2.yaml out/experiments/terrain_tests_v3/partial_pointcloud/no_wrench_v1 -m test -s
python scripts/eval_results.py cfg/dataset/sim_terrain_test_v2.yaml out/experiments/terrain_tests_v3/partial_pointcloud/forward_def_v1 -m test -s
