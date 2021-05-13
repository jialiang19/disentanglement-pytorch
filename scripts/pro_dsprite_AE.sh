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
--use_wandb=true \
--max_iter=115200 \
--max_epoch=115200 \
--evaluation_metric beta_vae_sklearn factor_vae_metric \
--num_worker=4 \
--all_iter=1000 \
--gpu_id=1 \
--inc_every_n_epoch=1 \

