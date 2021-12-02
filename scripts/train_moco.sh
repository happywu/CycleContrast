#!/bin/sh

OUTDIR='output/mocov2_res50_r2v2_ep200'
mkdir -p $OUTDIR

# lr and batch size are for 8-gpu

python train_cycle.py \
  -a resnet50 \
  --lr 0.06 \
  --moco-dim 128 \
  --batch-size 512 \
  --epochs 200 --schedule 120 160 \
  --save-dir $OUTDIR \
  --moco-t 0.07 \
  --dataset r2v2 \
  --world-size 1 \
  --rank 0 \
  --multiprocessing-distributed \
  --num-of-sampled-frames 1 \
  --dist-url 'tcp://localhost:10001' \
  --mlp --cos --aug-plus \
  data/r2v2_large_with_ids/train &> ${OUTDIR}/log_128pernode.txt
