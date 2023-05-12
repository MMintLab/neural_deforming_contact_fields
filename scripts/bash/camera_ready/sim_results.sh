# python scripts/generate.py cfg/primitives/camera_ready/model_v1.yaml -d cfg/primitives/camera_ready/dataset/test_v1.yaml -m test -o out/experiments/camera_ready/sim/test_v1/model_v1/ --gen_args "{"iter_limit": 100, "embed_weight": 0.001, "contact_threshold": 0.2, "mesh_resolution": 256}"

# python scripts/generate.py cfg/primitives/camera_ready/model_v1.yaml -d cfg/primitives/camera_ready/dataset/test_v1.yaml -m test -o out/experiments/camera_ready/sim/test_v1/model_v1_v2/ --gen_args "{"iter_limit": 100, "embed_weight": 0.001, "contact_threshold": 0.2}"

python scripts/generate.py cfg/primitives/camera_ready/model_v1.yaml -d cfg/primitives/camera_ready/dataset/test_v1.yaml -m test -o out/experiments/camera_ready/sim/test_v1/model_v1_v3/ --gen_args "{"iter_limit": 100, "embed_weight": 0.001, "contact_threshold": 0.2, "num_latent": 4}"
