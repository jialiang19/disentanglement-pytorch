#! /bin/sh

FILENAME=$(basename $0)
FILENAME="${FILENAME%.*}"
NAME=${1:-$FILENAME}

echo "name=$NAME"

python3 main.py \
--name=$NAME \
--alg=PROAE \
--dset_dir=$DISENTANGLEMENT_LIB_DATA  \
--dset_name=dsprites_full \
--encoder=ProConv64 \
--decoder=SimpleConv64 \
--z_dim=1 \
--w_recon=10000 \
--use_wandb=false \
--max_iter=11520 \
--max_epoch=11520 \
--num_worker=1 \
--all_iter=100 \
--gpu_id=0 \
--inc_every_n_epoch=0.1 \

