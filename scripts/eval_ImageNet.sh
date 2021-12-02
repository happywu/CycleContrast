#!/bin/sh

OUTDIR=$1
DATA=$2

# lr and batch size are for 8-gpu

python main_lincls.py \
  -a resnet50 \
  --lr 30 \
  --pretrained ${OUTDIR}/checkpoint_0199.pth.tar \
  --batch-size 256 \
  --save-dir $OUTDIR \
  --world-size 1 \
  --dataset imagenet \
  --rank 0 \
  --dist-url 'tcp://localhost:10001' \
  --multiprocessing-distributed \
 ${DATA} &> ${OUTDIR}/lincls_IN_ep100_lr30_bsz256.txt
