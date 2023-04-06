# python scripts/generate.py cfg/terrain_tests/wrench_v2.yaml -d cfg/dataset/real_terrain_test.yaml -m test -o out/experiments/real_terrain_test_v2/partial_pointcloud/wrench_v2/ --gen_args "{"embed_weight": 1.0, "def_loss": 0.0}"

python scripts/eval_results_real.py cfg/dataset/real_terrain_test.yaml out/experiments/real_terrain_test_v2/partial_pointcloud/wrench_v2 -m test -s
python scripts/eval_results_real.py cfg/dataset/real_terrain_test.yaml out/experiments/real_terrain_test_v2/partial_pointcloud/baseline -m test -s