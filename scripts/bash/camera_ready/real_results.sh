# python scripts/generate.py cfg/primitives/camera_ready/model_v1.yaml -d cfg/primitives/camera_ready/dataset/real_test_v1.yaml -m test -o out/experiments/camera_ready/real/model_v1/ --gen_args "{"iter_limit": 100, "embed_weight": 1.0, "contact_threshold": 0.8}"

python scripts/generate.py cfg/primitives/camera_ready/model_v3.yaml -d cfg/primitives/camera_ready/dataset/real_test_v1.yaml -m test -o out/experiments/camera_ready/real/model_v3/ --gen_args "{"iter_limit": 100, "embed_weight": 0.1, "contact_threshold": 0.2}"
