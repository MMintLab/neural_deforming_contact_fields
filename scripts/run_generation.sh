#python scripts/tune_contact_threshold.py cfg/wrench_v2_tests/wrench_v1.yaml cfg/dataset/wrench_v2_val.yaml -o out/experiments/wrench_v2_tests/tune_generation/wrench_v1/
#python scripts/tune_contact_threshold.py cfg/wrench_v2_tests/wrench_v2.yaml cfg/dataset/wrench_v2_val.yaml -o out/experiments/wrench_v2_tests/tune_generation/wrench_v2/
#python scripts/tune_contact_threshold.py cfg/wrench_v2_tests/wrench_v3.yaml cfg/dataset/wrench_v2_val.yaml -o out/experiments/wrench_v2_tests/tune_generation/wrench_v3/
#python scripts/tune_contact_threshold.py cfg/wrench_v2_tests/wrench_v4.yaml cfg/dataset/wrench_v2_val.yaml -o out/experiments/wrench_v2_tests/tune_generation/wrench_v4/
#python scripts/tune_contact_threshold.py cfg/wrench_v2_tests/wrench_v5.yaml cfg/dataset/wrench_v2_val.yaml -o out/experiments/wrench_v2_tests/tune_generation/wrench_v5/
#python scripts/tune_contact_threshold.py cfg/wrench_v2_tests/wrench_v6.yaml cfg/dataset/wrench_v2_val.yaml -o out/experiments/wrench_v2_tests/tune_generation/wrench_v6/
#python scripts/tune_contact_threshold.py cfg/wrench_v2_tests/wrench_v7.yaml cfg/dataset/wrench_v2_val.yaml -o out/experiments/wrench_v2_tests/tune_generation/wrench_v7/
#python scripts/tune_contact_threshold.py cfg/wrench_v2_tests/wrench_v8.yaml cfg/dataset/wrench_v2_val.yaml -o out/experiments/wrench_v2_tests/tune_generation/wrench_v8/
#python scripts/tune_contact_threshold.py cfg/wrench_v2_tests/no_wrench_v1.yaml cfg/dataset/wrench_v2_val.yaml -o out/experiments/wrench_v2_tests/tune_generation/no_wrench_v1/
#python scripts/tune_contact_threshold.py cfg/wrench_v2_tests/no_wrench_v2.yaml cfg/dataset/wrench_v2_val.yaml -o out/experiments/wrench_v2_tests/tune_generation/no_wrench_v2/
#python scripts/tune_contact_threshold.py cfg/wrench_v2_tests/forward_def_v1.yaml cfg/dataset/wrench_v2_val.yaml -o out/experiments/wrench_v2_tests/tune_generation/forward_def_v1/
#
#python scripts/generate.py cfg/wrench_v2_tests/wrench_v1.yaml -d cfg/dataset/wrench_v2_tests.yaml -m test -f model_threshold.pt -o out/experiments/wrench_v2_tests/partial_pointcloud/wrench_v1/
#python scripts/generate.py cfg/wrench_v2_tests/wrench_v2.yaml -d cfg/dataset/wrench_v2_tests.yaml -m test -f model_threshold.pt -o out/experiments/wrench_v2_tests/partial_pointcloud/wrench_v2/
#python scripts/generate.py cfg/wrench_v2_tests/wrench_v3.yaml -d cfg/dataset/wrench_v2_tests.yaml -m test -f model_threshold.pt -o out/experiments/wrench_v2_tests/partial_pointcloud/wrench_v3/
#python scripts/generate.py cfg/wrench_v2_tests/wrench_v4.yaml -d cfg/dataset/wrench_v2_tests.yaml -m test -f model_threshold.pt -o out/experiments/wrench_v2_tests/partial_pointcloud/wrench_v4/
#python scripts/generate.py cfg/wrench_v2_tests/wrench_v5.yaml -d cfg/dataset/wrench_v2_tests.yaml -m test -f model_threshold.pt -o out/experiments/wrench_v2_tests/partial_pointcloud/wrench_v5/
#python scripts/generate.py cfg/wrench_v2_tests/wrench_v6.yaml -d cfg/dataset/wrench_v2_tests.yaml -m test -f model_threshold.pt -o out/experiments/wrench_v2_tests/partial_pointcloud/wrench_v6/
#python scripts/generate.py cfg/wrench_v2_tests/wrench_v7.yaml -d cfg/dataset/wrench_v2_tests.yaml -m test -f model_threshold.pt -o out/experiments/wrench_v2_tests/partial_pointcloud/wrench_v7/
#python scripts/generate.py cfg/wrench_v2_tests/wrench_v8.yaml -d cfg/dataset/wrench_v2_tests.yaml -m test -f model_threshold.pt -o out/experiments/wrench_v2_tests/partial_pointcloud/wrench_v8/
#python scripts/generate.py cfg/wrench_v2_tests/no_wrench_v1.yaml -d cfg/dataset/wrench_v2_tests.yaml -m test -f model_threshold.pt -o out/experiments/wrench_v2_tests/partial_pointcloud/no_wrench_v1/
#python scripts/generate.py cfg/wrench_v2_tests/no_wrench_v2.yaml -d cfg/dataset/wrench_v2_tests.yaml -m test -f model_threshold.pt -o out/experiments/wrench_v2_tests/partial_pointcloud/no_wrench_v2/
#python scripts/generate.py cfg/wrench_v2_tests/forward_def_v1.yaml -d cfg/dataset/wrench_v2_tests.yaml -m test -f model_threshold.pt -o out/experiments/wrench_v2_tests/partial_pointcloud/forward_def_v1/
#
#python scripts/eval_results.py cfg/dataset/wrench_v2_tests.yaml out/experiments/wrench_v2_tests/partial_pointcloud/wrench_v1 -m test
#python scripts/eval_results.py cfg/dataset/wrench_v2_tests.yaml out/experiments/wrench_v2_tests/partial_pointcloud/wrench_v2 -m test
#python scripts/eval_results.py cfg/dataset/wrench_v2_tests.yaml out/experiments/wrench_v2_tests/partial_pointcloud/wrench_v3 -m test
#python scripts/eval_results.py cfg/dataset/wrench_v2_tests.yaml out/experiments/wrench_v2_tests/partial_pointcloud/wrench_v4 -m test
#python scripts/eval_results.py cfg/dataset/wrench_v2_tests.yaml out/experiments/wrench_v2_tests/partial_pointcloud/wrench_v5 -m test
#python scripts/eval_results.py cfg/dataset/wrench_v2_tests.yaml out/experiments/wrench_v2_tests/partial_pointcloud/wrench_v6 -m test
#python scripts/eval_results.py cfg/dataset/wrench_v2_tests.yaml out/experiments/wrench_v2_tests/partial_pointcloud/wrench_v7 -m test
#python scripts/eval_results.py cfg/dataset/wrench_v2_tests.yaml out/experiments/wrench_v2_tests/partial_pointcloud/wrench_v8 -m test
#python scripts/eval_results.py cfg/dataset/wrench_v2_tests.yaml out/experiments/wrench_v2_tests/partial_pointcloud/no_wrench_v1 -m test
#python scripts/eval_results.py cfg/dataset/wrench_v2_tests.yaml out/experiments/wrench_v2_tests/partial_pointcloud/no_wrench_v2 -m test
#python scripts/eval_results.py cfg/dataset/wrench_v2_tests.yaml out/experiments/wrench_v2_tests/partial_pointcloud/forward_def_v1 -m test

python scripts/generate.py cfg/wrench_v2_tests/wrench_v1.yaml -d cfg/dataset/wrench_v2_tests.yaml -m test -o out/experiments/wrench_v2_tests/partial_pointcloud/wrench_v1/
python scripts/generate.py cfg/wrench_v2_tests/wrench_v2.yaml -d cfg/dataset/wrench_v2_tests.yaml -m test -o out/experiments/wrench_v2_tests/partial_pointcloud/wrench_v2/
python scripts/generate.py cfg/wrench_v2_tests/wrench_v3.yaml -d cfg/dataset/wrench_v2_tests.yaml -m test -o out/experiments/wrench_v2_tests/partial_pointcloud/wrench_v3/
python scripts/generate.py cfg/wrench_v2_tests/wrench_v4.yaml -d cfg/dataset/wrench_v2_tests.yaml -m test -o out/experiments/wrench_v2_tests/partial_pointcloud/wrench_v4/
python scripts/generate.py cfg/wrench_v2_tests/wrench_v5.yaml -d cfg/dataset/wrench_v2_tests.yaml -m test -o out/experiments/wrench_v2_tests/partial_pointcloud/wrench_v5/
python scripts/generate.py cfg/wrench_v2_tests/wrench_v6.yaml -d cfg/dataset/wrench_v2_tests.yaml -m test -o out/experiments/wrench_v2_tests/partial_pointcloud/wrench_v6/
python scripts/generate.py cfg/wrench_v2_tests/wrench_v7.yaml -d cfg/dataset/wrench_v2_tests.yaml -m test -o out/experiments/wrench_v2_tests/partial_pointcloud/wrench_v7/
python scripts/generate.py cfg/wrench_v2_tests/wrench_v8.yaml -d cfg/dataset/wrench_v2_tests.yaml -m test -o out/experiments/wrench_v2_tests/partial_pointcloud/wrench_v8/
python scripts/generate.py cfg/wrench_v2_tests/no_wrench_v1.yaml -d cfg/dataset/wrench_v2_tests.yaml -m test -o out/experiments/wrench_v2_tests/partial_pointcloud/no_wrench_v1/
python scripts/generate.py cfg/wrench_v2_tests/no_wrench_v2.yaml -d cfg/dataset/wrench_v2_tests.yaml -m test -o out/experiments/wrench_v2_tests/partial_pointcloud/no_wrench_v2/
python scripts/generate.py cfg/wrench_v2_tests/forward_def_v1.yaml -d cfg/dataset/wrench_v2_tests.yaml -m test -o out/experiments/wrench_v2_tests/partial_pointcloud/forward_def_v1/

python scripts/generate.py cfg/wrench_v2_tests/wrench_v1.yaml -d cfg/dataset/wrench_v2_tests.yaml -m test -o out/experiments/wrench_v2_tests/partial_pointcloud_v2/wrench_v1/
python scripts/generate.py cfg/wrench_v2_tests/wrench_v2.yaml -d cfg/dataset/wrench_v2_tests.yaml -m test -o out/experiments/wrench_v2_tests/partial_pointcloud_v2/wrench_v2/
python scripts/generate.py cfg/wrench_v2_tests/wrench_v3.yaml -d cfg/dataset/wrench_v2_tests.yaml -m test -o out/experiments/wrench_v2_tests/partial_pointcloud_v2/wrench_v3/
python scripts/generate.py cfg/wrench_v2_tests/wrench_v4.yaml -d cfg/dataset/wrench_v2_tests.yaml -m test -o out/experiments/wrench_v2_tests/partial_pointcloud_v2/wrench_v4/
python scripts/generate.py cfg/wrench_v2_tests/wrench_v5.yaml -d cfg/dataset/wrench_v2_tests.yaml -m test -o out/experiments/wrench_v2_tests/partial_pointcloud_v2/wrench_v5/
python scripts/generate.py cfg/wrench_v2_tests/wrench_v6.yaml -d cfg/dataset/wrench_v2_tests.yaml -m test -o out/experiments/wrench_v2_tests/partial_pointcloud_v2/wrench_v6/
python scripts/generate.py cfg/wrench_v2_tests/wrench_v7.yaml -d cfg/dataset/wrench_v2_tests.yaml -m test -o out/experiments/wrench_v2_tests/partial_pointcloud_v2/wrench_v7/
python scripts/generate.py cfg/wrench_v2_tests/wrench_v8.yaml -d cfg/dataset/wrench_v2_tests.yaml -m test -o out/experiments/wrench_v2_tests/partial_pointcloud_v2/wrench_v8/
python scripts/generate.py cfg/wrench_v2_tests/no_wrench_v1.yaml -d cfg/dataset/wrench_v2_tests.yaml -m test -o out/experiments/wrench_v2_tests/partial_pointcloud_v2/no_wrench_v1/
python scripts/generate.py cfg/wrench_v2_tests/no_wrench_v2.yaml -d cfg/dataset/wrench_v2_tests.yaml -m test -o out/experiments/wrench_v2_tests/partial_pointcloud_v2/no_wrench_v2/
python scripts/generate.py cfg/wrench_v2_tests/forward_def_v1.yaml -d cfg/dataset/wrench_v2_tests.yaml -m test -o out/experiments/wrench_v2_tests/partial_pointcloud_v2/forward_def_v1/
