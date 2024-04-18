#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Give me a number"
    exit 1
fi

num=$1

source_file_drag="/home/dhlab/Desktop/452 Final Project/model_checkpoints/drag_${num}.ckpt"
source_file_lift="/home/dhlab/Desktop/452 Final Project/model_checkpoints/lift_${num}.ckpt"
source_file_momentum="/home/dhlab/Desktop/452 Final Project/model_checkpoints/momentum_${num}.ckpt"
source_file_discriminator="/home/dhlab/Desktop/452 Final Project/model_checkpoints/discriminator_${num}.ckpt"
source_file_generator="/home/dhlab/Desktop/452 Final Project/model_checkpoints/generator_${num}.ckpt"

# check that the files exist

if [ ! -f "$source_file_drag" ]; then
    echo "$source_file_drag does not exist"
    exit 2
fi

if [ ! -f "$source_file_lift" ]; then
    echo "$source_file_lift does not exist"
    exit 2
fi

if [ ! -f "$source_file_momentum" ]; then
    echo "$source_file_momentum does not exist"
    exit 2
fi

if [ ! -f "$source_file_discriminator" ]; then
    echo "$source_file_discriminator does not exist"
    exit 2
fi

if [ ! -f "$source_file_generator" ]; then
    echo "$source_file_generator does not exist"
    exit 2
fi


cp "$source_file_drag" "$2"
cp "$source_file_lift" "$2"
cp "$source_file_momentum" "$2"
cp "$source_file_discriminator" "$2"
cp "$source_file_generator" "$2"