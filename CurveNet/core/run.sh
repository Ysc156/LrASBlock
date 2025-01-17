#!/bin/bash
    #python3 main_partseg.py --exp_name=curveunet_seg_skip_new --epoch=200
    python3 main_partseg.py --exp_name=curveunet_seg_skip_new --eval=True --model_path=../checkpoints/curveunet_seg_skip_new/models/model.t7