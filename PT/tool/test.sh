#!/bin/sh

export PYTHONPATH=./
eval "$(conda shell.bash hook)"
eval "conda activate pt"
PYTHON=python

TEST_CODE=test.py

dataset=$1
exp_name=$2
exp_dir=exp/${dataset}_new/${exp_name}
model_dir=${exp_dir}/model
result_dir=${exp_dir}/result
#config=config/${dataset}/${dataset}_${exp_name}.yaml
config=config/${dataset}/${dataset}_new.yaml
mkdir -p ${result_dir}/last
mkdir -p ${result_dir}/best

now=$(date +"%Y%m%d_%H%M%S")
cp ${config} tool/test.sh tool/${TEST_CODE} ${exp_dir}


#: '
$PYTHON -u ${exp_dir}/${TEST_CODE} \
  --config=${config} \
  save_folder ${result_dir}/best \
  model_path ${model_dir}/model_best.pth \
  2>&1 | tee ${exp_dir}/test_best-$now.log
#'

