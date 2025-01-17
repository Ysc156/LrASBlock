#!/bin/bash
#while [ 1 ];do
    # python3 train.py --config config/s3dis/s3dis_stratified_transformer.yaml
#    python3 train.py --config config/s3dis/s3dis_swin3d_transformer.yaml 
#    CUDA_VISIBLE_DEVICES=5 python3 test.py --config config/s3dis/s3dis_stratified_transformer.yaml
   python3 test.py --config config/s3dis/s3dis_stratified_transformer.yaml
   python3 train_s.py --config config/scannetv2/scannetv2_S_TF.yaml
   python3 train.py --config config/kitti/kitti_S_TF.yaml
#done
