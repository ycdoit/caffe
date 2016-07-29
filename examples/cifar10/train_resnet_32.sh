#!/usr/bin/env sh

TOOLS=./build/tools

$TOOLS/caffe train \
    --solver=examples/cifar10/cifar10_resnet_32_solver.prototxt

# reduce learning rate by factor of 10
$TOOLS/caffe train \
    --solver=examples/cifar10/cifar10_resnet_32_solver_lr1.prototxt \
    --snapshot=examples/cifar10/cifar10_resnet_32_iter_32000.solverstate.h5

# reduce learning rate by factor of 10
$TOOLS/caffe train \
    --solver=examples/cifar10/cifar10_resnet_32_solver_lr2.prototxt \
    --snapshot=examples/cifar10/cifar10_resnet_32_iter_48000.solverstate.h5
