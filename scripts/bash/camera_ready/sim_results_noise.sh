#python scripts/generate.py cfg/primitives/camera_ready/model_v3.yaml -d cfg/primitives/camera_ready/dataset/noise/noise_0.1.yaml -m test -o out/experiments/camera_ready/sim_noise/model_v3/noise_0.1/ --gen_args "{"iter_limit": 100, "embed_weight": 0.001, "contact_threshold": 0.2}"
#python scripts/generate.py cfg/primitives/camera_ready/model_v3.yaml -d cfg/primitives/camera_ready/dataset/noise/noise_0.5.yaml -m test -o out/experiments/camera_ready/sim_noise/model_v3/noise_0.5/ --gen_args "{"iter_limit": 100, "embed_weight": 0.001, "contact_threshold": 0.2}"
#python scripts/generate.py cfg/primitives/camera_ready/model_v3.yaml -d cfg/primitives/camera_ready/dataset/noise/noise_1.0.yaml -m test -o out/experiments/camera_ready/sim_noise/model_v3/noise_1.0/ --gen_args "{"iter_limit": 100, "embed_weight": 0.001, "contact_threshold": 0.2}"
#
#python scripts/eval_results.py cfg/primitives/camera_ready/dataset/noise/noise_0.1.yaml out/experiments/camera_ready/sim_noise/model_v3/noise_0.1/ -m test -s
#python scripts/eval_results.py cfg/primitives/camera_ready/dataset/noise/noise_0.5.yaml out/experiments/camera_ready/sim_noise/model_v3/noise_0.5/ -m test -s
#python scripts/eval_results.py cfg/primitives/camera_ready/dataset/noise/noise_1.0.yaml out/experiments/camera_ready/sim_noise/model_v3/noise_1.0/ -m test -s

python scripts/plot/plot_inf_params_single.py out/experiments/camera_ready/sim_noise/model_v3/noise_0.1/ out/experiments/camera_ready/sim_noise/model_v3/noise_0.5/ out/experiments/camera_ready/sim_noise/model_v3/noise_1.0/ out/experiments/camera_ready/sim_noise/model_v3/noise_res.csv -n noise_0.1 noise_0.5 noise_1.0
