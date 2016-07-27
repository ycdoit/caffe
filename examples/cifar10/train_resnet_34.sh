#!/usr/bin/env sh

TOOLS=./build/tools

$TOOLS/caffe train \
    --solver=examples/cifar10/cifar10_resnet_34_solver.prototxt

# reduce learning rate by factor of 10
$TOOLS/caffe train \
    --solver=examples/cifar10/cifar10_resnet_34_solver_lr1.prototxt \
    --snapshot=examples/cifar10/cifar10_resnet_34_iter_16000.solverstate.h5

# reduce learning rate by factor of 10
$TOOLS/caffe train \
    --solver=examples/cifar10/cifar10_resnet_34_solver_lr2.prototxt \
    --snapshot=examples/cifar10/cifar10_resnet_34_iter_24000.solverstate.h5
