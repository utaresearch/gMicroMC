#ifndef GLOBALKERNEL_CUH
#define GLOBALKERNEL_CUH
#include "global.h"

__device__ int applyBoundary(int shape, float3 center, float3 size, float x, float y, float z);

__device__ void rotate(gFloat *u, gFloat *v, gFloat *w, gFloat costh, gFloat phi);
__device__ void rotate(float3 &direction, float costh, float phi);
__global__ void setupcuseed(int num, int* iseed1, curandState *cuseed);
void iniCuseed(int *iseed1);

#endif
