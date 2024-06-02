#!/bin/bash
datasets=('eurosat.yaml' 'food101.yaml' 'dtd.yaml' 'fgvc.yaml' 'oxford_flowers.yaml' 'oxford_pets.yaml' 'stanford_cars.yaml' 'ucf101.yaml' 'caltech101.yaml' 'sun397.yaml' 'imagenet.yaml' )
seeds=('1' '2' '3' '4' '5')

architecture="$2"
GPU="$1"
logs="$3"

if [ -z "$architecture" ]; then
    architecture="Vit-B16"
fi

if [ -z "$logs" ]; then
    logs="./results"
fi

if [ -z "$GPU" ]; then
    GPU="0"
fi

export CUDA_VISIBLE_DEVICES=$GPU

for dataset in "${datasets[@]}"
do
    for seed in "${seeds[@]}"
    do
        echo "python main.py --config configs/decoop/$architecture.yaml --dataset configs/datasets/$dataset --seed $seed"
        python main.py --config configs/decoop/$architecture.yaml --dataset configs/datasets/$dataset --seed $seed --logdir $logs
    done
done