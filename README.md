## Contrastive Learning of Image Representations with Cross-Video Cycle-Consistency



This is a official implementation of the CycleContrast introduced in the paper:[**Contrastive Learning of Image Representations with Cross-Video Cycle-Consistency**](https://arxiv.org/pdf/2105.06463.pdf)
### Citation
If you find our work useful, please cite:
```
@article{wu2021contrastive,
  title={Contrastive Learning of Image Representations with Cross-Video Cycle-Consistency},
  author={Wu, Haiping and Wang, Xiaolong},
  journal={arXiv preprint arXiv:2105.06463},
  year={2021}
}
```

### Preparation

Our code is tested on Python 3.7 and Pytorch 1.3.0, please install the environment via 

```
pip install -r requirements.txt
```

### Model Zoo 

We provide the model pretrained on R2V2 for 200 epochs. 

| method        | pre-train epochs on R2V2 dataset | ImageNet Top-1 Linear Eval | OTB Precision | OTB Success | UCF Top-1 | pretrained model |
|---------------|:--------------------------------:|:--------------------------:|:-------------:|:-----------:|:---------:|------------------|
| MoCo          |                200               |            53.8            |      56.1     |     40.6    |    80.5   |                  |
| CycleContrast |                200               |            55.7            |      69.6     |     50.4    |    82.8   |                  |

### Run Experiments 

#### Data preparation

Download R2V2 (Random Related Video Views) dataset according to https://github.com/danielgordon10/vince.

The direction structure should be as followed:
```
CycleContrast
├── cycle_contrast 
├── scripts 
├── utils 
├── data
│   ├── r2v2_large_with_ids 
│   │   ├── train 
│   │   │   ├── --/
│   │   │   ├── -_/
│   │   │   ├── _-/
│   │   │   ├── __/
│   │   │   ├── -0/
│   │   │   ├── _0/
│   │   │   ├── ...
│   │   │   ├── zZ/
│   │   │   ├── zz/
│   │   ├── val
│   │   │   ├── --/
│   │   │   ├── -_/
│   │   │   ├── _-/
│   │   │   ├── __/
│   │   │   ├── -0/
│   │   │   ├── _0/
│   │   │   ├── ...
│   │   │   ├── zZ/
│   │   │   ├── zz/
```

#### Unsupervised Pretrain

```
./scripts/train_cycle.sh
```


#### Downstream task - ImageNet linear eval

Prepare ImageNet dataset according to [pytorch ImageNet training code](https://github.com/pytorch/examples/tree/master/imagenet).

```shell
MODEL_DIR=output/cycle_res50_r2v2_ep200
IMAGENET_DATA=data/ILSVRC/Data/CLS-LOC
./scripts/eval_ImageNet.sh $MODEL_DIR $IMAGENET_DATA
```

### Downstream task - OTB tracking

Transfer to OTB tracking evaluation is based on [SiamFC-Pytorch](https://github.com/huanglianghua/siamfc-pytorch). 
Please prepare environment and data according to [SiamFC-Pytorch](https://github.com/huanglianghua/siamfc-pytorch)

```shell
git clone https://github.com/happywu/mmaction2-CycleContrast
# path to your pretrained model, change accordingly
CycleContrast=/home/user/code/CycleContrast
PRETRAIN=${CycleContrast}/output/cycle_res50_r2v2_ep200/checkpoint_0199.pth.tar
cd mmaction2_tracking
./scripts/submit_r2v2_r50_cycle.py ${PRETRAIN}
```

### Downstream task - UCF classification

Transfer to UCF action recognition evaluation is based on [AVID-CMA](https://github.com/facebookresearch/AVID-CMA), 
prepare data and env according to [AVID-CMA](https://github.com/facebookresearch/AVID-CMA).

```shell
git clone https://github.com/happywu/AVID-CMA-CycleContrast
# path to your pretrained model, change accordingly
CycleContrast=/home/user/code/CycleContrast
PRETRAIN=${CycleContrast}/output/cycle_res50_r2v2_ep200/checkpoint_0199.pth.tar
cd AVID-CMA-CycleContrast 
./scripts/submit_r2v2_r50_cycle.py ${PRETRAIN}
```


## Acknowledgements

The codebase is based on [FAIR-MoCo](https://github.com/facebookresearch/moco).
The OTB tracking evaluation is based on [MMAction2](https://github.com/open-mmlab/mmaction2), [SiamFC-PyTorch](https://github.com/huanglianghua/siamfc-pytorch) and [vince](https://github.com/danielgordon10/vince).
The UCF classification evaluation follows [AVID-CMA](https://github.com/facebookresearch/AVID-CMA).

Thank you all for the great open source repositories!
