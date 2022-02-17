#ifndef GLOBAL_H
#define GLOBAL_H

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#include <cuda.h>
#include <curand_kernel.h>

#include "rapidjson/document.h"
using namespace rapidjson;

#define gFloat float

// //functions for checking
#define CUDA_CALL(x) do{if(x != cudaSuccess){printf("CUDA Error at %s:%d\n",__FILE__,__LINE__);exit(1);}}while(0)
int applyROISearch(int shape, float3 center, float3 size, float x, float y, float z);
// global variable used by all functions
extern Document document;
extern int verbose, NPART, NRAND, NTHREAD_PER_BLOCK, deviceIndex;

extern cudaDeviceProp devProp;
extern __device__ curandState *cuseed;

// constants
#define MCC 510998.9461 // rest mass energy ~511keV
#define TWOMCC 1021997.8922 //2*MC^2
#define M2C4 261119922915.31070521 //M^2C^4
#define PI 3.14159265359
#define C 299792458
#define ZERO 1.0e-20
#define SZERO 1.0e-6

#endif