# UV-Relighting
UV map relighting based on input environment maps. For a captured 3D subject, we recover its mesh and UV map. We then investigate different approaches for relighting an input full light diffuse UV map under different illuminations. We condition it against various methods namely
+ Channel Conditioning
+ Latent variable conditioning
+ SIPR based conditioning
+ Layer conditioning.

The network is then extended to predict per Gaussian features such as position,scale,color and opacity.
# Instructions to run
```
python train.py \
--train_uv_maps <train_uv_maps> \
--val_uv_maps <val_uv_maps> \
--train_env_maps <train_env_maps> \
--val_env_maps <val_env_maps> \
--diffuse <input_diffuse_map> \
--save_dir <save_dir> \
--lr 1e-3 \
--train \
--network <network>
```