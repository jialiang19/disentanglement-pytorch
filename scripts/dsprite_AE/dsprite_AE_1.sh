#! /bin/sh

FILENAME=$(basename $0)
FILENAME="${FILENAME%.*}"
NAME=${1:-$FILENAME}

echo "name=$NAME"

python3 main.py \
--name=$NAME \
--alg=AE \
--dset_dir=$DISENTANGLEMENT_LIB_DATA  \
--dset_name=dsprites_full \
--encoder=SimpleConv64 \
--decoder=SimpleConv64 \
--z_dim=10 \
--w_recon=10000 \
--use_wandb=true \
--max_epoch=115200 \
--max_iter=115200 \
--evaluation_metric beta_vae_sklearn factor_vae_metric \
--num_worker=1 \
--all_iter=3840 \
--gpu_id=1 \





