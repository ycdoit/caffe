#ifndef CAFFE_UTIL_GPU_ALLOCATOR_H_
#define CAFFE_UTIL_GPU_ALLOCATOR_H_

#include <set>
#include <vector>
#include <string>

namespace caffe {

class GpuAllocator {
 public:
    bool GetGpus(std::vector<int>* gpus, int count);
 private:
  std::vector<std::string> allocated_gpus;
  void TouchGPULockFile(std::string file_path);
  std::string GetHostName();
  void ReadAllLines(std::string file_path, std::set<std::string>* lines);
  void AppendLine(std::string file_path, std::string line);
  const char* lock_file_name = "gpu_allocator.lock";
  const char* gpu_allocation_file_name = "gpu_allocation.content";
};
}  // namespace caffe

#endif
