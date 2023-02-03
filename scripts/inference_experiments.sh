#python scripts/generate.py cfg/terrain_tests/wrench_v1.yaml -d cfg/dataset/real_tool_test_v1.yaml -m test -o out/experiments/real_world_test/test_loss_weights/exp_v1/ --gen_args "{"embed_weight": 0.01, "def_loss": 0.0}"
#python scripts/generate.py cfg/terrain_tests/wrench_v1.yaml -d cfg/dataset/real_tool_test_v1.yaml -m test -o out/experiments/real_world_test/test_loss_weights/exp_v2/ --gen_args "{"embed_weight": 0.1, "def_loss": 0.0}"
#python scripts/generate.py cfg/terrain_tests/wrench_v1.yaml -d cfg/dataset/real_tool_test_v1.yaml -m test -o out/experiments/real_world_test/test_loss_weights/exp_v3/ --gen_args "{"embed_weight": 1.0, "def_loss": 0.0}"
#python scripts/generate.py cfg/terrain_tests/wrench_v1.yaml -d cfg/dataset/real_tool_test_v1.yaml -m test -o out/experiments/real_world_test/test_loss_weights/exp_v4/ --gen_args "{"embed_weight": 10.0, "def_loss": 0.0}"
#python scripts/generate.py cfg/terrain_tests/wrench_v1.yaml -d cfg/dataset/real_tool_test_v1.yaml -m test -o out/experiments/real_world_test/test_loss_weights/exp_v5/ --gen_args "{"embed_weight": 100.0, "def_loss": 0.0}"
#python scripts/generate.py cfg/terrain_tests/wrench_v1.yaml -d cfg/dataset/real_tool_test_v1.yaml -m test -o out/experiments/real_world_test/test_loss_weights/exp_v6/ --gen_args "{"embed_weight": 1000.0, "def_loss": 0.0}"

python scripts/eval_results_real.py cfg/dataset/real_tool_test_v1.yaml out/experiments/real_world_test/test_loss_weights/exp_v1/ -m test
python scripts/eval_results_real.py cfg/dataset/real_tool_test_v1.yaml out/experiments/real_world_test/test_loss_weights/exp_v2/ -m test
python scripts/eval_results_real.py cfg/dataset/real_tool_test_v1.yaml out/experiments/real_world_test/test_loss_weights/exp_v3/ -m test
python scripts/eval_results_real.py cfg/dataset/real_tool_test_v1.yaml out/experiments/real_world_test/test_loss_weights/exp_v4/ -m test
python scripts/eval_results_real.py cfg/dataset/real_tool_test_v1.yaml out/experiments/real_world_test/test_loss_weights/exp_v5/ -m test
python scripts/eval_results_real.py cfg/dataset/real_tool_test_v1.yaml out/experiments/real_world_test/test_loss_weights/exp_v6/ -m test
