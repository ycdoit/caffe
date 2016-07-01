#include <string>
#include <vector>

#include "caffe/sgd_solvers.hpp"
#include "caffe/util/hdf5.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/upgrade_proto.hpp"
#include <multiverso/multiverso.h>
#include <multiverso/updater/updater.h>
#include <multiverso/updater/momentum_updater.h>

namespace caffe {

template <typename Dtype>
void ASGDSolver<Dtype>::ASGDPreSolve() {
    auto callback = new ASGDCallback();
    callback->SetSolver(this);
    this->add_callback(callback);
    auto param_size = GetParamSize();
    size_ = param_size;
    params_1.reset(new Blob<Dtype>({ param_size, 1, 1, 1 }));
    params_2.reset(new Blob<Dtype>({ param_size, 1, 1, 1 }));
    worker_table.reset(new multiverso::ArrayWorker<Dtype>(param_size));
    server_table.reset(new multiverso::ArrayServer<Dtype>(param_size));
    multiverso::MV_Barrier();

    params_train = params_1;
    // init the pipeline buffer
    CopyModelToBuffer(params_train);
    SubmitModelToServer(params_train);

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
    return size;
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

template <typename Dtype>
void ASGDSolver<Dtype>::SubmitModelToServer(boost::shared_ptr<Blob<Dtype>> buffer) {
    multiverso::MV_Barrier();
    if (multiverso::MV_Rank() == 0)
    {
        SubmitModelToServer(buffer->mutable_cpu_data(), size_);//send the initial params to server
    }
    multiverso::MV_Barrier();
}

template <typename Dtype>
void ASGDSolver<Dtype>::SubmitModelToServer(Dtype* model, int size) {
    boost::shared_ptr<Dtype> current_model(new Dtype[size], [](Dtype *p) { delete[] p; });
    // clear model
    worker_table->Get(current_model.get(), size);
    multiverso::AddOption option;
    option.set_momentum(0);
    worker_table->Add(current_model.get(), size, &option);
    for (int i = 0; i < size; i++)
    {
        current_model.get()[i] = 0.0f;
    }
    // clear momentum
    worker_table->Add(current_model.get(), size, &option);

    // replace with new model
    for (auto i = 0; i < size; ++i)       model[i] = -1 * model[i];
    worker_table->Add(model, size, &option);

    for (int i = 0; i < size; i++)
    {
        current_model.get()[i] = 0.0f;
    }
    // clear momentum
    worker_table->Add(current_model.get(), size, &option);
}

template <typename Dtype>
void ASGDSolver<Dtype>::CopyModelToBuffer(boost::shared_ptr<Blob<Dtype>> buffer) {
    int count = 0;

    // TODO(junli): fix this.
   /* switch (Caffe::mode()) {
    case Caffe::CPU:
        blob_mem_ptr = (data_or_diff == DATA ? blob->cpu_data() : blob->cpu_diff());
        break;
    case Caffe::GPU:
        blob_mem_ptr = (data_or_diff == DATA ? blob->gpu_data() : blob->gpu_diff());
        break;
    default:
        mxERROR("Unknown Caffe mode");
    }
    caffe_copy(blob->count(), blob_mem_ptr, mat_mem_ptr);*/

    for (int param_id = 0; param_id < this->net_->learnable_params().size(); ++param_id) {
        auto weight = this->net_->learnable_params()[param_id];
        caffe_copy(weight->count(), weight->gpu_data(), buffer->mutable_gpu_data() + count);
        count += weight->count();
    }
}

template <typename Dtype>
void ASGDSolver<Dtype>::CopyBufferToModel(boost::shared_ptr<Blob<Dtype>> buffer) {
    int count = 0;

    // TODO(junli): fix this.
  /*  switch (Caffe::mode()) {
    case Caffe::CPU:
        blob_mem_ptr = (data_or_diff == DATA ? blob->cpu_data() : blob->cpu_diff());
        break;
    case Caffe::GPU:
        blob_mem_ptr = (data_or_diff == DATA ? blob->gpu_data() : blob->gpu_diff());
        break;
    default:
        mxERROR("Unknown Caffe mode");
    }
    caffe_copy(blob->count(), blob_mem_ptr, mat_mem_ptr);
*/

    for (int param_id = 0; param_id < this->net_->learnable_params().size(); ++param_id) {
        auto weight = this->net_->learnable_params()[param_id];
        caffe_copy(weight->count(), buffer->gpu_data() + count, weight->mutable_gpu_data());
        count += weight->count();
    }
}

template <typename Dtype>
void ASGDSolver<Dtype>::OnIterStart() {
    params_train = (*async_buffer->Get());
    CopyBufferToModel(params_train);
    BroadCastData();
}

// TODO(junli): Should support multiple Gpus per node
template <typename Dtype>
void ASGDSolver<Dtype>::BroadCastData() {
}

template <typename Dtype>
void ASGDSolver<Dtype>::GetModelFromServer(Dtype* model, int size) {
    worker_table->Get(model, size);
}

template <typename Dtype>
void ASGDSolver<Dtype>::SubmitDiffToServer(Dtype* model, int size) {
    multiverso::AddOption option;
    option.set_momentum(0.9f);                   // TODO(junli): Use parameter from proto file
    worker_table->Add(model, size, &option);
}

INSTANTIATE_CLASS(ASGDSolver);
REGISTER_SOLVER_CLASS(ASGD);

};