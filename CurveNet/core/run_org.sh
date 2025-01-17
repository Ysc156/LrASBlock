#!/bin/bash
#while [ 1 ];do
    #python3 partseg.py --exp_name=curveunet_seg_org_seed1 --epoch=200
    python3 partseg.py --exp_name=curveunet_seg_org_seed1 --eval=True --model_path=../checkpoints/curveunet_seg_org_seed1/models/model.t7