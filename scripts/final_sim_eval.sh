#python scripts/generate.py cfg/terrain_tests/wrench_v2.yaml -d cfg/dataset/sim_terrain_test_v2.yaml -m test -o out/experiments/terrain_tests_v2/partial_pointcloud/wrench_v2/ --gen_args "{"embed_weight": 1.0, "def_loss": 0.0}"
#python scripts/eval_results.py cfg/dataset/sim_terrain_test_v2.yaml out/experiments/terrain_tests_v2/partial_pointcloud/wrench_v2 -m test

#python scripts/eval_results.py cfg/dataset/sim_terrain_test_v2.yaml out/experiments/terrain_tests_v2/partial_pointcloud/baseline -m test

python scripts/generate.py cfg/terrain_tests/wrench_v2.yaml -d cfg/dataset/sim_terrain_test_v2.yaml -m test -o out/experiments/terrain_tests_v2/partial_pointcloud_vis/wrench_v2/ --gen_args "{"embed_weight": 1.0, "def_loss": 0.0}"
