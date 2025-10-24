Enhanced Underwater Object Detection via Multi-Scale Attention and Adaptive Feature Fusion
=
This is the code for this paper

Environment
=
```

conda create -n deim python=3.10
conda activate deim
pip install -r requirements.txt

```

Train
=
```

CUDA_VISIBLE_DEVICES=0 python train.py -c configs/yaml/deim_dfine_hgnetv2_s_mg.yml --seed=0

```

Test
=
```

python train.py -c configs/yaml/dfine_hgnetv2_s_mg.yml --test-only -r outputs/best_stg2.pth

```
Model
=
```

engine/extre_module/custom_nn/block/MANet
engine/extre_module/custom_nn/downsample/AWDS
engine/extre_module/custom_nn/featurefusion/CGFM
engine/extre_module/custom_nn/module/MSCB
engine/extre_module/custom_nn/transformer/LSSA

```
