#!/bin/bash

# ====== ENV VARS ======
# export TRANSFORMERS_CACHE='/ssddata/model_hub'
# export HF_DATASETS_CACHE='/ssddata/model_hub'
# export PYTORCH_KERNEL_CACHE_PATH='/ssddata/shiqi/error_tracing_4.31.0'
# export CUDA_VISIBLE_DEVICES="6"

# ====== CONFIG ======
dataset="hotpotqa"
data_path="../scripts/data/hpqa"  # nq → ../scripts/data/nq, trqa → ../scripts/data/trqa
model="daryl149/llama-2-7b-chat-hf"
model_name="${model#*/}"
debug=True

# model → (early_exit_layers, info_layer)
declare -A model_params
model_params["llama-2-7b-chat-hf"]="0,2,4,6,8,10,12,14,32|32"
model_params["llama-2-13b-chat-hf"]="0,2,4,6,8,10,12,14,16,18,40|40"
model_params["llama-2-70b-chat-hf"]="0,2,4,6,8,10,12,14,16,18,80|80"

# Parse params
IFS="|" read -r early_exit_layers info_layer <<< "${model_params[$model_name]}"

# ====== RUNS ======
echo "Model name: $model_name"
echo "Early exit layers: $early_exit_layers"
echo "Info layer: $info_layer"

## BASELINE
decoding_mode="baseline"
output_path="../res/res_hpqa/${model_name}/${model_name}_${decoding_mode}.json"
python ../eval_knowledge_qa.py \
    --model-name $model --dataset_name $dataset \
    --decoding_mode $decoding_mode \
    --output-path $output_path \
    --num-gpus 1 --do-rating --debug $debug

## DOLA
decoding_mode="dola"
output_path="../res/res_hpqa/${model_name}/${model_name}_${decoding_mode}.json"
python ../eval_knowledge_qa.py \
    --model-name $model --dataset_name $dataset \
    --decoding_mode $decoding_mode \
    --early_exit_layers $early_exit_layers \
    --output-path $output_path \
    --num-gpus 1 --do-rating --debug $debug

## ACTIVATION
decoding_mode="activation"
alpha="0.8"
decoding_strategy="entropy"  # or "single_entropy"
output_path="../res/res_hpqa/${model_name}/${model_name}_${decoding_mode}_${alpha}_${info_layer}.json"
python ../eval_knowledge_qa.py \
    --model-name $model --dataset_name $dataset \
    --decoding_mode $decoding_mode \
    --alpha $alpha --info_layer $info_layer --decoding_strategy $decoding_strategy \
    --output-path $output_path \
    --num-gpus 1 --do-rating --data_path $data_path --debug $debug

## ACTIVATION_DOLA
decoding_mode="activation_dola"
alpha="0.8"
decoding_strategy="entropy"
output_path="../res/res_hpqa/${model_name}/${model_name}_${decoding_mode}_${alpha}_${info_layer}.json"
python ../eval_knowledge_qa.py \
    --model-name $model --dataset_name $dataset \
    --decoding_mode $decoding_mode \
    --early_exit_layers $early_exit_layers \
    --alpha $alpha --info_layer $info_layer --decoding_strategy $decoding_strategy \
    --output-path $output_path \
    --num-gpus 1 --do-rating --data_path $data_path --debug $debug


## ACTIVATION_PROPERNOUN
