# NeRV
Neural Explicit Radiance Volumes

```
env CUDA_VISIBLE_DEVICES='4,5,6,7' python main_nerv1.py --accelerator='gpu' --devices=4 --batch_size=1 --lr=1e-4 --epochs=201 --logsdir=logs_nerv1 --datadir=data --train_samples=4000 --val_samples=800 --n_pts_per_ray=400 --vol_shape=256 --img_shape=256 --alpha=1 --theta=1 --gamma=1 --delta=1 --omega=0  --sh=3 --pe=3 --amp --vol
```