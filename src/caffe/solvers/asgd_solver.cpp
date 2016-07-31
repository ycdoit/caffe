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
void ASGDSolver<Dtype>::DebugModel(string str, const Blob<Dtype>& blob) {
  LOG(INFO) << str << " data[0]:" << blob.cpu_data()[0]
    << ",diff[0]:" << blob.cpu_diff()[0];
}

template <typename Dtype>
void ASGDSolver<Dtype>::ASGDPreSolve() {
  use_pipeline_ = false;
  LOG(INFO) << "Use Pipeline:" << use_pipeline_;
  auto callback = new ASGDCallback();
  callback->SetSolver(this);
  this->add_callback(callback);
  auto param_size = GetParamSize();
  size_ = param_size;
  vector<int> sizes = { param_size, 1, 1, 1 };
  params_1.reset(new Blob<Dtype>(sizes));
  params_2.reset(new Blob<Dtype>(sizes));
  worker_table.reset(new multiverso::ArrayWorker<Dtype>(param_size));
  server_table.reset(new multiverso::ArrayServer<Dtype>(param_size));
  multiverso::MV_Barrier();

  params_train = params_1;
  // init the pipeline buffer
  CopyModelToBuffer(params_train);
  SubmitModelToServer(params_train);

  const auto worker = this->worker_table;
  const auto size = param_size;
  if (use_pipeline_) {
    async_buffer = boost::shared_ptr<
      multiverso::ASyncBuffer<boost::shared_ptr<Blob<Dtype>>>>(
      new multiverso::ASyncBuffer<boost::shared_ptr<Blob<Dtype>>>(&params_1,
      &params_2,
      [worker, size](boost::shared_ptr<Blob<Dtype>>* buffer) -> void{
      worker->Get((*buffer)->mutable_cpu_data(), size);
      if (Caffe::mode() == Caffe::GPU) {
        (*buffer)->gpu_data();
      }
    }));
  }
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
    CHECK(Caffe::root_solver());
    Dtype rate = this->GetLearningRate();
    if (this->param_.display() && this->iter_ % this->param_.display() == 0) {
        LOG(INFO) << "Iteration " << this->iter_ << ", lr = " << rate;
    }
    this->ClipGradients();
    for (int param_id = 0; param_id < this->net_->learnable_params().size();
        ++param_id) {
        this->Normalize(param_id);
        this->Regularize(param_id);
        this->ComputeUpdateValue(param_id, rate);
    }

    // Copy diff to param_train, Submit diff to server
    CopyDiffToBuffer(params_train);
    SubmitDiffToServer(params_train->mutable_cpu_diff(), size_);
    // SubmitModelToServer(params_train);
}

template <typename Dtype>
void ASGDSolver<Dtype>::Snapshot() {
  if (multiverso::MV_Rank() == 0) {
    Solver<Dtype>::Snapshot();
  }
}

template <typename Dtype>
void ASGDSolver<Dtype>::SubmitModelToServer(
  boost::shared_ptr<Blob<Dtype>> buffer) {
    multiverso::MV_Barrier();
    if (multiverso::MV_Rank() == 0) {
        SubmitModelToServer(buffer->cpu_data(), size_);
    }
    multiverso::MV_Barrier();
}

template <typename Dtype>
void ASGDSolver<Dtype>::SubmitModelToServer(const Dtype* model, int size) {
    boost::shared_ptr<Dtype> current_model(
      new Dtype[size], [](Dtype *p) { delete[] p; });
    // clear model
    worker_table->Get(current_model.get(), size);
    multiverso::AddOption option;
    option.set_momentum(0);
    worker_table->Add(current_model.get(), size, &option);
    for (int i = 0; i < size; i++) {
        current_model.get()[i] = 0.0f;
    }
    // clear momentum
    worker_table->Add(current_model.get(), size, &option);

    // replace with new model
    for (auto i = 0; i < size; ++i) {
      current_model.get()[i] = -1 * model[i];
    }
    worker_table->Add(current_model.get(), size, &option);

    for (int i = 0; i < size; i++) {
        current_model.get()[i] = 0.0f;
    }
    // clear momentum
    worker_table->Add(current_model.get(), size, &option);


    worker_table->Get(current_model.get(), size_);
    CHECK(fabs(model[0] - current_model.get()[0]) <= 1e-10);
}

template <typename Dtype>
void ASGDSolver<Dtype>::CopyModelToBuffer(
  boost::shared_ptr<Blob<Dtype>> buffer) {
    int count = 0;
    for (int param_id = 0;
      param_id < this->net_->learnable_params().size();
      ++param_id) {
        auto weight = this->net_->learnable_params()[param_id];
        const Dtype* src = nullptr;
        Dtype* dst = nullptr;
        switch (Caffe::mode()) {
        case Caffe::CPU:
            src = weight->cpu_data();
            dst = buffer->mutable_cpu_data() + count;
            break;
        case Caffe::GPU:
            src = weight->gpu_data();
            dst = buffer->mutable_gpu_data() + count;
            break;
        default:
            LOG(ERROR) << "Unknown Caffe mode";
        }
        caffe_copy(weight->count(), src, dst);
        count += weight->count();
    }
}

template <typename Dtype>
void ASGDSolver<Dtype>::CopyDiffToBuffer(
  boost::shared_ptr<Blob<Dtype>> buffer) {
    int count = 0;
    for (int param_id = 0;
      param_id < this->net_->learnable_params().size();
      ++param_id) {
        auto weight = this->net_->learnable_params()[param_id];
        const Dtype* src = nullptr;
        Dtype* dst = nullptr;
        switch (Caffe::mode()) {
        case Caffe::CPU:
            src = weight->cpu_diff();
            dst = buffer->mutable_cpu_diff() + count;
            break;
        case Caffe::GPU:
            src = weight->gpu_diff();
            dst = buffer->mutable_gpu_diff() + count;
            break;
        default:
            LOG(ERROR) << "Unknown Caffe mode";
        }
        caffe_copy(weight->count(), src, dst);
        count += weight->count();
    }
}

template <typename Dtype>
void ASGDSolver<Dtype>::CopyBufferToModel(
  boost::shared_ptr<Blob<Dtype>> buffer) {
    int count = 0;
    for (int param_id = 0;
      param_id < this->net_->learnable_params().size();
      ++param_id) {
        auto weight = this->net_->learnable_params()[param_id];
        const Dtype* src = nullptr;
        Dtype* dst = nullptr;
        switch (Caffe::mode()) {
        case Caffe::CPU:
            src = buffer->cpu_data() + count;
            dst = weight->mutable_cpu_data();
            break;
        case Caffe::GPU:
            src = buffer->gpu_data() + count;
            dst = weight->mutable_gpu_data();
            break;
        default:
            LOG(ERROR) << "Unknown Caffe mode";
        }
        caffe_copy(weight->count(), src, dst);
        count += weight->count();
    }
}

#ifndef CPU_ONLY
template <typename Dtype>
void sgd_update_gpu(int N, Dtype* g, Dtype* h, Dtype momentum,
  Dtype local_rate);
#endif

template <typename Dtype>
void ASGDSolver<Dtype>::ComputeUpdateValue(int param_id, Dtype rate) {
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  const vector<float>& net_params_lr = this->net_->params_lr();
  // set local momentum = 0
  Dtype momentum = 0;
  Dtype local_rate = rate * net_params_lr[param_id];
  // Compute the update to history, then copy it to the parameter diff.
  switch (Caffe::mode()) {
  case Caffe::CPU: {
    caffe_cpu_axpby(net_params[param_id]->count(), local_rate,
      net_params[param_id]->cpu_diff(), momentum,
      history_[param_id]->mutable_cpu_data());
    caffe_copy(net_params[param_id]->count(),
      history_[param_id]->cpu_data(),
      net_params[param_id]->mutable_cpu_diff());
    break;
  }
  case Caffe::GPU: {
#ifndef CPU_ONLY
    sgd_update_gpu(net_params[param_id]->count(),
      net_params[param_id]->mutable_gpu_diff(),
      history_[param_id]->mutable_gpu_data(),
      momentum, local_rate);
#else
    NO_GPU;
#endif
    break;
  }
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
}

template <typename Dtype>
void ASGDSolver<Dtype>::OnIterStart() {
  if (use_pipeline_) {
    params_train = (*async_buffer->Get());
  } else {
    worker_table->Get(params_train->mutable_cpu_data(), size_);
  }
  CopyBufferToModel(params_train);
  BroadCastData();
}

template <typename Dtype>
void ASGDSolver<Dtype>::BroadCastData() {
  // multiple gpu broadcast is automatically supported by caffe::P2PSync
}

template <typename Dtype>
void ASGDSolver<Dtype>::GetModelFromServer(Dtype* model, int size) {
    worker_table->Get(model, size);
}

template <typename Dtype>
void ASGDSolver<Dtype>::SubmitDiffToServer(Dtype* model, int size) {
    multiverso::AddOption option;
    // TODO(junli): Use parameter from proto file
    option.set_momentum(this->param_.momentum());
    worker_table->Add(model, size, &option);
}

INSTANTIATE_CLASS(ASGDSolver);
REGISTER_SOLVER_CLASS(ASGD);
};  // namespace caffe
