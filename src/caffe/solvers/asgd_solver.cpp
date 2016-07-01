#include <string>
#include <vector>

#include "caffe/sgd_solvers.hpp"
#include "caffe/util/hdf5.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/upgrade_proto.hpp"
#include "multiverso/multiverso.h"

namespace caffe {

template <typename Dtype>
void ASGDSolver<Dtype>::ASGDPreSolve() {
    auto callback = new ASGDCallback();
    callback->SetSolver(this);
    this->add_callback(callback);
    auto param_size = GetParamSize();
    params_1.reset(new Blob<Dtype>({ param_size, 1, 1, 1 }));
    params_2.reset(new Blob<Dtype>({ param_size, 1, 1, 1 }));
    worker_table.reset(new multiverso::ArrayWorker<Dtype>(param_size));
    server_table.reset(new multiverso::ArrayServer<Dtype>(param_size));
    multiverso::MV_Barrier();

    params_train = params_1;
    // TODO(junli): init server buffer.
    // init the pipeline buffer
    //copy_model_to_buffer(params_train);
    //SubmitModelToServer(params_train);

    const auto worker = this->worker_table;
    const auto size = param_size;
    async_buffer = boost::shared_ptr<multiverso::ASyncBuffer<boost::shared_ptr<Blob<Dtype>>>>(new multiverso::ASyncBuffer<boost::shared_ptr<Blob<Dtype>>>(&params_1,
        &params_2,
        [worker, size](boost::shared_ptr<Blob<Dtype>>* buffer) -> void{
        worker->Get((*buffer)->mutable_cpu_data(), size);
        (*buffer)->gpu_data();
    }));
}

template <typename Dtype>
int ASGDSolver<Dtype>::GetParamSize() {
    auto size = 0;
    for (int param_id = 0; param_id < this->net_->learnable_params().size();
        ++param_id) {
        size += this->net_->learnable_params()[param_id]->count();
    }
    return 0;
}


template <typename Dtype>
void ASGDSolver<Dtype>::ApplyUpdate() {

}

template <typename Dtype>
void ASGDSolver<Dtype>::Snapshot() {
  if (multiverso::MV_Rank() == 0) {
    Solver<Dtype>::Snapshot();
  }
}

INSTANTIATE_CLASS(ASGDSolver);
REGISTER_SOLVER_CLASS(ASGD);

};