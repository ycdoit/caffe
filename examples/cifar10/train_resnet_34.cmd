"Build/x64/Debug/caffe.exe" train --solver=examples/cifar10/cifar10_resnet_34_solver.prototxt

REM reduce learning rate by factor of 10
"Build/x64/Debug/caffe.exe" train --solver=examples/cifar10/cifar10_resnet_34_solver_lr1.prototxt --snapshot=examples/cifar10/cifar10_resnet_34_iter_16000.solverstate.h5

REM reduce learning rate by factor of 10
"Build/x64/Debug/caffe.exe" train --solver=examples/cifar10/cifar10_resnet_34_solver_lr2.prototxt --snapshot=examples/cifar10/cifar10_resnet_34_iter_24000.solverstate.h5
