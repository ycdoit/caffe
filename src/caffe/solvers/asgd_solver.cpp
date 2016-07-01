#include <string>
#include <vector>

#include "caffe/sgd_solvers.hpp"
#include "caffe/util/hdf5.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/upgrade_proto.hpp"

namespace caffe {

template <typename Dtype>
void ASGDSolver<Dtype>::ASGDPreSolve() {

}


template <typename Dtype>
void ASGDSolver<Dtype>::ApplyUpdate() {

}

INSTANTIATE_CLASS(ASGDSolver);
REGISTER_SOLVER_CLASS(ASGD);

};