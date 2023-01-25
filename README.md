# Neural Deforming Contact Fields

Model, training, and inference code for *Neural Deforming Contact Fields* project.

## Setup

Using conda:
```bash
conda create -f environment.yaml
conda activate ncf
```

Our code also relies on the following libraries. Install each to the `ncf` environment using `pip install`:

* [mmint_utils](https://github.com/MMintLab/mmint_utils)
* [pytorch-meta](https://github.com/tristandeleu/pytorch-meta)

## Data

Datasets contain full simulated data including geometry, contact patches and 
force responses, generated with Isaac Gym.

* Training dataset: download [here](https://drive.google.com/file/d/1m1dpCBkz0Qwjwus-FDfhDgqk-AvqhUhv/view?usp=sharing) (115.9 MB)
* Test dataset: download [here](https://drive.google.com/file/d/1RzXtE_fRF4_taVZzP2lA5_6XGHF4nM31/view?usp=share_link) (10.3 MB)

Unzip dataset(s) into `out/datasets/`.


## Training

Model training is split into a *pretraining* and *training* step. Training options are specified by
`yaml` config files. See `cfg/example_v1.yaml` for an example config. You can specify which
dataset to use for pretraining/training, model choice and hyper-parameters, loss weights, etc.

### Pretraining

```
python neural_contact_fields/pretrain_model.py cfg/example_v1.yaml
```

### Training

```
python neural_contact_fields/train_model.py cfg/example_v1.yaml
```

## Inference

### Training Data Tests

To test how the model performs on pretraining data:
```
python scripts/train/test_object_module_pretraining.py cfg/example_v1.yaml -v
```

To test how model performs on training data:
```
python scripts/train/test_inference_latent.py cfg/example_v1.yaml -v -m train
```

### Inference tests:

To run inference tests on testing data, we pass a config file specifying the test dataset with the `-d` option.
Removing will test inference procedures on training data.

To test inferring latent variables given *full* supervision:
```
python scripts/inference/infer_latent.py cfg/example_v1.yaml -d cfg/dataset/example_gen.yaml
```

To test inferring latent variables given *surface samples*:
```
python scripts/inference/infer_latent_from_surface.py cfg/example_v1.yaml -d cfg/dataset/example_gen.yaml
```

## Running Baseline
### Setup

```angular2html
conda env create -f environment_b1.yml
conda activate ncf_b1
```
Our code also relies on the following libraries. Install each to the `ncf` environment using `pip install`:
* [mmint_utils](https://github.com/MMintLab/mmint_utils)

Note: Please do not build torchmeta.

### Clone baseline backbone
``` 
git clone https://github.com/hzxie/GRNet neural_contact_fields/explicit_baseline/grnet
```
### Installation
Please **skip** ``pip install -r `` step and jump to building extensions in [readme](https://github.com/hzxie/GRNet#prerequisites) via ``pip install -e .``.
[Important] Please make sure that nvcc is installed via ``$ sudo apt install nvidia-cuda-toolkit`` in your system before building extensions.


### Troubleshooting
* If you get 
``
OSError: /home/young/.local/lib/python3.8/site-packages/torch/lib/../../nvidia/cublas/lib/libcublas.so.11: undefined symbol: cublasLtHSHMatmulAlgoInit, version libcublasLt.so.11
``
do
``$ export LD_LIBRARY_PATH=/home/myenv/.local/lib/python3.8/site-packages/nvidia/cublas/lib/
``
* If you have trouble with ```openexr``` installation
``` 
conda install -c conda-forge cudatoolkit-dev 
```
and
```angular2html
sudo apt-get install openexr
sudo apt-get install libopenexr-dev
```


* When you meet CUB compatibility error as follows:
```angular2html
#error The version of CUB in your include path is not compatible with this release of Thrust. CUB is now included in the CUDA Toolkit, so you no longer need to use your own checkout of CUB. Define THRUST_IGNORE_CUB_VERSION_CHECK to ignore this.
```
, go to ``` '~/miniconda3/envs/ncf/include/thrust/system/cuda/config.h' ``` and add this line
``` #define THRUST_IGNORE_CUB_VERSION_CHECK true ```  before ```#ifndef THRUST_IGNORE_CUB_VERSION_CHECK```.
