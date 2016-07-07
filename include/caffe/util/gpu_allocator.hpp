#ifndef CAFFE_UTIL_DEVICE_ALTERNATE_H_
#define CAFFE_UTIL_DEVICE_ALTERNATE_H

#include <vector>
using std::vector;

namespace caffe {

class GpuAllocator
{
public:
    static bool GetGpus(vector<int>* gpus, int count);
};

}

#endif