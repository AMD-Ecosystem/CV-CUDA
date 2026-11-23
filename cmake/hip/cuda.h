#pragma once
#include "CvCudaHipCompat.h"

// The few CUDA driver API names the tests use to inspect a captured graph's
// kernel launch geometry. HIP reports the same data through the runtime as
// hipKernelNodeParams, whose block and grid sizes are dim3 members rather than
// separate scalar fields, so the driver struct is rebuilt from it.
#if defined(__cplusplus)
typedef hipError_t CUresult;
#    define CUDA_SUCCESS hipSuccess

struct CUDA_KERNEL_NODE_PARAMS
{
    unsigned int gridDimX, gridDimY, gridDimZ;
    unsigned int blockDimX, blockDimY, blockDimZ;
    unsigned int sharedMemBytes;
};

inline CUresult cuGraphKernelNodeGetParams(hipGraphNode_t node, CUDA_KERNEL_NODE_PARAMS *params)
{
    hipKernelNodeParams hipParams{};
    hipError_t          err = hipGraphKernelNodeGetParams(node, &hipParams);
    if (err == hipSuccess)
    {
        params->gridDimX       = hipParams.gridDim.x;
        params->gridDimY       = hipParams.gridDim.y;
        params->gridDimZ       = hipParams.gridDim.z;
        params->blockDimX      = hipParams.blockDim.x;
        params->blockDimY      = hipParams.blockDim.y;
        params->blockDimZ      = hipParams.blockDim.z;
        params->sharedMemBytes = hipParams.sharedMemBytes;
    }
    return err;
}
#endif
