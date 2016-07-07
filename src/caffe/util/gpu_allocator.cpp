#include <boost/interprocess/sync/named_mutex.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <map>
#include <string>

#include "caffe/util/gpu_allocator.hpp"
#include <multiverso/multiverso.h>
#include "caffe/common.hpp"

using namespace std;
using namespace boost::interprocess;


namespace caffe {

bool GpuAllocator::GetGpus(vector<int>* gpus, int count) {
    int to_allocate_count = count;
#ifndef CPU_ONLY 
    string name_start = "gpu-";
    int device_count = 0;
    auto mpi_rank = multiverso::MPI_RANK();
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    for (auto i = 0; i < device_count && to_allocate_count > 0; ++i) {
        auto gpu_name = name_start + to_string(i);
        try {
            named_mutex mutex(create_only, gpu_name);
        }
        catch(interprocess_exception &ex) {
            LOG(INFO) << "Lock " << gpu_name << " failed for process " << mpi_rank ;
            continue;
        }
        LOG(INFO) << "Lock " << gpu_name << " success for process " << mpi_rank ;
        gpus->push(i);
        --to_allocate_count;
    }
#endif

    return to_allocate_count == 0;
}

}  // namespace caffe