#!/usr/bin/env sh

TOOLS=./build/tools

mpiexec -n 4 $TOOLS/caffe asgd_train \
    --solver=examples/cifar10/cifar10_full_solver_4p4G1M.prototxt \
    --gpus_per_process=1