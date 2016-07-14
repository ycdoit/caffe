#include "caffe/util/gpu_allocator.hpp"
#include "caffe/common.hpp"

#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <boost/interprocess/sync/file_lock.hpp>
#include <boost/filesystem.hpp>
#include <boost/thread/lock_guard.hpp>
#include <boost/asio/detail/socket_ops.hpp>

#include <multiverso/multiverso.h>

using std::string;
using std::vector;
using std::set;
using std::endl;
using boost::system::error_code;
using boost::system::system_category;

namespace caffe {

void GpuAllocator::TouchGPULockFile(string file_path) {
  auto rank = multiverso::MV_Rank();
  if (rank == 0) {
    std::ofstream f(file_path);
    CHECK(f) << "Create GPU lock file failed.";
    LOG(INFO) << "Created GPU lock file.";
  }
  multiverso::MV_Barrier();
}

string GpuAllocator::GetHostName() {
  char name[1024];
  boost::system::error_code ec;
  CHECK(!boost::asio::detail::socket_ops::gethostname(name, sizeof(name), ec))
    << "Get host name failed for rank "
    << multiverso::MV_Rank();
  return std::string(name);
}

void GpuAllocator::ReadAllLines(string file_path, set<string>* lines) {
  std::string line;
  std::ifstream in_file(file_path);
  CHECK(in_file) << "Read " << file_path << "error." << endl;
  while (std::getline(in_file, line)) {
    if (line.length() > 0) {
      lines->insert(line);
    }
  }
  in_file.close();
  LOG(INFO) << "Read " << lines->size() << " lines from " << file_path
    << "from rank " << multiverso::MV_Rank();
}

void GpuAllocator::AppendLine(string file_path, const string line) {
  std::ofstream out_file(file_path, std::ios_base::app);
  out_file << line << endl;
  out_file.flush();
  out_file.close();

  LOG(INFO) << "Wrote 1 line to " << file_path
    << "from rank " << multiverso::MV_Rank();
}

// Allocates GPUs to current process.
// Note: we use file lock to allocate GPUs exclusively. A file locking is a
// class that has process lifetime. This means that if a process holding a file
// lock ends or crashes, the operating system will automatically unlock it.
bool GpuAllocator::GetGpus(vector<int>* gpus, int count) {
  TouchGPULockFile(gpu_allocation_file_name);
  auto hostName = GetHostName();

  int to_allocate_count = count;
#ifndef CPU_ONLY
  string name_start = hostName + ":gpu-";
  int device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  boost::interprocess::file_lock flock(lock_file_name);
  {
    boost::lock_guard<boost::interprocess::file_lock> guard(flock);
    set<string> allocated_gpus;
    for (auto i = 0; i < device_count && to_allocate_count > 0; ++i) {
      ReadAllLines(gpu_allocation_file_name, &allocated_gpus);
      string gpu_name = name_start + std::to_string(i);
      if (allocated_gpus.find(gpu_name) == allocated_gpus.end()) {
        allocated_gpus.insert(gpu_name);
        AppendLine(gpu_allocation_file_name, gpu_name);
        LOG(INFO) << "Lock " << gpu_name << " success for process "
          << multiverso::MV_Rank();
        gpus->push_back(i);
        --to_allocate_count;
      } else {
        LOG(INFO) << "Lock " << gpu_name << " failed for process "
          << multiverso::MV_Rank();
      }
    }
  }
#endif
  return to_allocate_count == 0;
}

}  // namespace caffe
