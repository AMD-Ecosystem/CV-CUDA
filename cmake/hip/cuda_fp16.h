#pragma once
#include "CvCudaHipCompat.h"
#include <hip/hip_fp16.h>

// CUDA's __double2half rounds the double straight to half. HIP has no such function,
// and going through float would round twice, so convert through _Float16, which
// clang narrows from double in one correctly rounded step.
#if defined(__cplusplus) && defined(__HIPCC__)
__host__ __device__ inline __half __double2half(const double x)
{
    return __half(static_cast<_Float16>(x));
}
#endif
