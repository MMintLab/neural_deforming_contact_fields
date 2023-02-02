python scripts/generate.py cfg/terrain_tests/wrench_v1.yaml -d cfg/dataset/real_tool_test_v1.yaml -m test -o out/experiments/real_world_test/test_ati_variations/exp_v1/ --gen_args "{"embed_weight": 0.001, "def_loss": 1.0}"
python scripts/generate.py cfg/terrain_tests/wrench_v1.yaml -d cfg/dataset/real_tool_test_v1.yaml -m test -o out/experiments/real_world_test/test_ati_variations/exp_v2/ --gen_args "{"embed_weight": 0.01, "def_loss": 1.0}"
python scripts/generate.py cfg/terrain_tests/wrench_v1.yaml -d cfg/dataset/real_tool_test_v1.yaml -m test -o out/experiments/real_world_test/test_ati_variations/exp_v3/ --gen_args "{"embed_weight": 0.1, "def_loss": 1.0}"

python scripts/generate.py cfg/terrain_tests/wrench_v1.yaml -d cfg/dataset/real_tool_test_v1.yaml -m test -o out/experiments/real_world_test/test_ati_variations/exp_v4/ --gen_args "{"embed_weight": 0.001, "def_loss": 10.0}"
python scripts/generate.py cfg/terrain_tests/wrench_v1.yaml -d cfg/dataset/real_tool_test_v1.yaml -m test -o out/experiments/real_world_test/test_ati_variations/exp_v5/ --gen_args "{"embed_weight": 0.01, "def_loss": 10.0}"
python scripts/generate.py cfg/terrain_tests/wrench_v1.yaml -d cfg/dataset/real_tool_test_v1.yaml -m test -o out/experiments/real_world_test/test_ati_variations/exp_v6/ --gen_args "{"embed_weight": 0.1, "def_loss": 10.0}"

python scripts/generate.py cfg/terrain_tests/wrench_v1.yaml -d cfg/dataset/real_tool_test_v1.yaml -m test -o out/experiments/real_world_test/test_ati_variations/exp_v7/ --gen_args "{"embed_weight": 0.001, "def_loss": 100.0}"
python scripts/generate.py cfg/terrain_tests/wrench_v1.yaml -d cfg/dataset/real_tool_test_v1.yaml -m test -o out/experiments/real_world_test/test_ati_variations/exp_v8/ --gen_args "{"embed_weight": 0.01, "def_loss": 100.0}"
python scripts/generate.py cfg/terrain_tests/wrench_v1.yaml -d cfg/dataset/real_tool_test_v1.yaml -m test -o out/experiments/real_world_test/test_ati_variations/exp_v9/ --gen_args "{"embed_weight": 0.1, "def_loss": 100.0}"
