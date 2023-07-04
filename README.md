# Neural Deforming Contact Fields

Model, training, and inference code for *Neural Deforming Contact Fields* project.

## Setup

Using conda:
```bash
conda env create -f environment.yaml
conda activate ncf
```

Our code also relies on the following libraries. Install each to the `ncf` environment using `pip install`:

* [mmint_utils](https://github.com/MMintLab/mmint_utils)
* [pytorch-meta](https://github.com/tristandeleu/pytorch-meta)

## Data/Models

Datasets contain full simulated data including geometry, contact patches and 
force responses, generated with Isaac Gym. Code to generate data can be found [here](https://github.com/MMintLab/ndcf_envs).

* Pretrain dataset: download [here](https://www.dropbox.com/s/ygp3lz09gn1183m/nominal_0.pkl.gzip?dl=0) (1.4 MB)
* Training dataset: download [here](https://www.dropbox.com/s/bllhkpm1f6fknrq/combine_train_proc.zip?dl=0) (26 GB)
* Test dataset: download [here](https://www.dropbox.com/s/2yopvlgudax3t8c/combine_test_proc.zip?dl=0) (2.6 GB)

The final model used in our experiments can be downloaded from the following places.

* Pretrained object model: [here](https://www.dropbox.com/s/81woclhrgejtb43/pretrain_v4.zip?dl=0) (0.63 GB)
* Full model: [here](https://www.dropbox.com/s/ah2uqormcx8fcmq/model_v3.zip?dl=0) (1.7 GB).

See `cfg/example_v1.yaml` for expected locations in order to run with the pretrained models.

## Training

Model training is split into a *pretraining* and *training* step. Training options are specified by
`yaml` config files. See `cfg/example_v1.yaml` for an example config. You can specify which
dataset to use for pretraining/training, model choice and hyper-parameters, loss weights, etc.

### Pretraining

```
python scripts/pretrain_model.py cfg/example_v1.yaml
```

### Training

```
python scripts/train_model.py cfg/example_v1.yaml
```

## Inference

To generate results:
```
python scripts/generate.py cfg/example_v1.yaml -m test -o <out dir>
```

To visualize results:
```
python scripts/vis_results.py cfg/example_v1.yaml -m test <out dir>
```

To evaluate results:
```
python scripts/eval_results.py cfg/example_v1.yaml -m test <out dir>
```