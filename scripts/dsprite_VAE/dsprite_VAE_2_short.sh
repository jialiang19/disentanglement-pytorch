#! /bin/sh

FILENAME=$(basename $0)
FILENAME="${FILENAME%.*}"
NAME=${1:-$FILENAME}

echo "name=$NAME"

python3 main.py \
--name=$NAME \
--alg=BetaVAE \
--dset_dir=$DISENTANGLEMENT_LIB_DATA  \
--dset_name=dsprites_full \
--traverse_z=true \
--encoder=SimpleGaussianConv64 \
--decoder=SimpleConv64 \
--z_dim=10 \
--w_kld=4 \
--use_wandb=true \
--max_epoch=11520 \
--max_iter=11520 \
--evaluation_metric beta_vae_sklearn factor_vae_metric \
--num_worker=1 \
--all_iter=384 \
--gpu_id=-1 \
--inc_every_n_epoch=0.1 \


