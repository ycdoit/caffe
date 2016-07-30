#!/usr/bin/env sh

TOOLS=./build/tools

$TOOLS/caffe asgd_train \
    --solver=examples/cifar10/cifar10_full_solver_1p4G0M.prototxt \
    --gpus_per_process=1