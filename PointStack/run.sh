#!/bin/bash
#while [ 1 ];do
#  python train.py --cfg_file 'cfgs/partnormal/pointstack.yaml' --exp_name 'experiments' --val_steps 1
  python test.py --cfg_file 'cfgs/partnormal/pointstack.yaml' --ckpt 'experiments/PartNormal/experiments/ckpt/ckpt-best.pth'
#
#done
