#python scripts/generate.py cfg/terrain_tests/wrench_v1.yaml -d cfg/dataset/sim_terrain_test.yaml -m test -o out/experiments/terrain_tests/partial_pointcloud/wrench_v1/ --gen_args "{"embed_weight": 1.0, "def_loss": 0.0}"
#python scripts/generate.py cfg/terrain_tests/wrench_v2.yaml -d cfg/dataset/sim_terrain_test.yaml -m test -o out/experiments/terrain_tests/partial_pointcloud/wrench_v2/ --gen_args "{"embed_weight": 1.0, "def_loss": 0.0}"
#python scripts/generate.py cfg/terrain_tests/wrench_v3.yaml -d cfg/dataset/sim_terrain_test.yaml -m test -o out/experiments/terrain_tests/partial_pointcloud/wrench_v3/ --gen_args "{"embed_weight": 1.0, "def_loss": 0.0}"
#python scripts/generate.py cfg/terrain_tests/wrench_v4.yaml -d cfg/dataset/sim_terrain_test.yaml -m test -o out/experiments/terrain_tests/partial_pointcloud/wrench_v4/ --gen_args "{"embed_weight": 1.0, "def_loss": 0.0}"
#
#python scripts/eval_results.py cfg/dataset/sim_terrain_test.yaml out/experiments/terrain_tests/partial_pointcloud/wrench_v1 -m test
#python scripts/eval_results.py cfg/dataset/sim_terrain_test.yaml out/experiments/terrain_tests/partial_pointcloud/wrench_v2 -m test
#python scripts/eval_results.py cfg/dataset/sim_terrain_test.yaml out/experiments/terrain_tests/partial_pointcloud/wrench_v3 -m test
#python scripts/eval_results.py cfg/dataset/sim_terrain_test.yaml out/experiments/terrain_tests/partial_pointcloud/wrench_v4 -m test
#python scripts/eval_results.py cfg/dataset/sim_terrain_test.yaml out/experiments/terrain_tests/partial_pointcloud/test_final -m test
#
#python scripts/generate.py cfg/terrain_tests/wrench_v1.yaml -d cfg/dataset/real_terrain_test.yaml -m test -o out/experiments/real_terrain_test/partial_pointcloud/wrench_v1/ --gen_args "{"embed_weight": 1.0, "def_loss": 0.0}"
#python scripts/generate.py cfg/terrain_tests/wrench_v2.yaml -d cfg/dataset/real_terrain_test.yaml -m test -o out/experiments/real_terrain_test/partial_pointcloud/wrench_v2/ --gen_args "{"embed_weight": 1.0, "def_loss": 0.0}"
#python scripts/generate.py cfg/terrain_tests/wrench_v3.yaml -d cfg/dataset/real_terrain_test.yaml -m test -o out/experiments/real_terrain_test/partial_pointcloud/wrench_v3/ --gen_args "{"embed_weight": 1.0, "def_loss": 0.0}"
#python scripts/generate.py cfg/terrain_tests/wrench_v4.yaml -d cfg/dataset/real_terrain_test.yaml -m test -o out/experiments/real_terrain_test/partial_pointcloud/wrench_v4/ --gen_args "{"embed_weight": 1.0, "def_loss": 0.0}"

#python scripts/eval_results_real.py cfg/dataset/real_terrain_test.yaml out/experiments/real_terrain_test/partial_pointcloud/wrench_v1 -m test
#python scripts/eval_results_real.py cfg/dataset/real_terrain_test.yaml out/experiments/real_terrain_test/partial_pointcloud/wrench_v2 -m test
#python scripts/eval_results_real.py cfg/dataset/real_terrain_test.yaml out/experiments/real_terrain_test/partial_pointcloud/wrench_v3 -m test
#python scripts/eval_results_real.py cfg/dataset/real_terrain_test.yaml out/experiments/real_terrain_test/partial_pointcloud/wrench_v4 -m test

python scripts/generate.py cfg/terrain_tests/wrench_v2.yaml -d cfg/dataset/sim_terrain_test.yaml -m test -o out/experiments/terrain_tests/partial_pointcloud_vis/wrench_v2/ --gen_args "{"embed_weight": 1.0, "def_loss": 0.0}"