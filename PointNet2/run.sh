#!/bin/bash
#while [ 1 ];do
#    python train_partseg.py --model pointnet2_part_seg_msg --normal --log_dir pointnet2_part_seg_msg
#    python train_partseg.py --model pointnet2_part_seg_msg_softmax --normal --log_dir pointnet2_part_seg_msg_softmax
#    python train_partseg.py --model pointnet2_part_seg_msg_r1 --normal --log_dir pointnet2_part_seg_msg_r1
#    python train_partseg.py --model pointnet2_part_seg_TD_msg --normal --log_dir pointnet2_part_seg_TD_msg
#    python train_partseg.py --model PCT --normal --log_dir PCT
    python train_partseg.py --model point --normal --log_dir point
    python test_partseg.py --normal --log_dir point
#    python test_partseg.py --normal --log_dir pointnet2_part_seg_msg
#done
