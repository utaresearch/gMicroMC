#ifndef PHYSICSKERNEL_CUH
#define PHYSICSKERNEL_CUH

#include <thrust/device_vector.h> 
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

#include "global.h"
#include "physicsList.h"

extern __constant__ float eEcutoff, pECutoff;
extern __constant__ float3 boundaryCenter, boundarySize, ROICenter, ROISize;
extern __constant__ int boundaryShape, ROIShape;
template<typename T>
 struct sortPtypeDescend
 {
   typedef T first_argument_type;
 
   typedef T second_argument_type;
 
   typedef bool result_type;
 
   __thrust_exec_check_disable__
   __host__ __device__ bool operator()(const T &lhs, const T &rhs) const {return lhs.ptype > rhs.ptype;}
 };
template<typename T>
 struct sortEDescend
 {
   typedef T first_argument_type;
 
   typedef T second_argument_type;
 
   typedef bool result_type;
 
   __thrust_exec_check_disable__
   __host__ __device__ bool operator()(const T &lhs, const T &rhs) const {return lhs.e > rhs.e;}
 };

void sampleSource(int num, int ptype, float A, float R, float *h_eneprob,Particle *h_particles);
void sortEofElectron(Particle *dev_e2Queue, int sim_num_elec);

__device__ float fetchData(cudaTextureObject_t CSTable, int i);
__global__ void printTex(cudaTextureObject_t CSTable);
void runPrint(cudaTextureObject_t CSTable);

__global__ void setParticles(int num, int ptype, float A, float RMAX, Particle* d_eQueue);
__global__  void eTransport(int N, int ContainerSize, Particle *eQueue, Particle *e2Queue, Data *container,
          int *where, int *second_num, int *gEid,
		  cudaTextureObject_t DACSTable, cudaTextureObject_t BindE_array, 
		  cudaTextureObject_t ieeCSTable, cudaTextureObject_t elastDCSTable, int MaxN);
__global__  void pTransport(int N, int ContainerSize, Particle *eQueue, Particle *e2Queue, Data *container, int *where, int *second_num, int *gEid,
    cudaTextureObject_t protonCSTable, int MaxN);

__device__ void getDAcs(gFloat e, gFloat *cs, gFloat *valid, cudaTextureObject_t DACSTable);
__device__ void getElastCS(gFloat elog, gFloat *cs, gFloat *valid, cudaTextureObject_t ieeCSTable);
__device__ void getExcitCS(gFloat elog, gFloat *cs, gFloat *valid, cudaTextureObject_t ieeCSTable);
__device__ void getIonCS(gFloat elog, gFloat *cs, gFloat *valid, cudaTextureObject_t ieeCSTable);
__device__ void tableIntp(gFloat elog, Particle *electron, gFloat *reacCS, gFloat *reacValid, cudaTextureObject_t ieeCSTable, cudaTextureObject_t DACSTable);

__device__ gFloat eDistance(curandState *seed, gFloat csSum);
__device__ void actChoice(curandState *seed, gFloat *reacValid, gFloat *reacCS, Particle *electron_ptr);
__device__ gFloat wx(gFloat r, gFloat t, uint c);
__device__ gFloat hx(gFloat t, gFloat w, gFloat tp, uint c);
__device__ void ionE2nd(gFloat * e2nd, Particle * thisOne, uint channel, curandState *seed);
__device__ void eDrop(curandState *seed, Particle *electron_ptr, gFloat elog, gFloat *ei, gFloat *e2nd, gFloat *eRatio, cudaTextureObject_t BindE_array, float *edrop);

__device__ gFloat elastDCS(curandState *seed, gFloat elog, cudaTextureObject_t elastDCSTable);
__device__ void eAngles(curandState *seed, Particle *electron_ptr, gFloat elog, gFloat ei, gFloat e2nd, gFloat eRatio, gFloat *polar, gFloat *azi, gFloat *polar_e2nd, gFloat *azi_e2nd, cudaTextureObject_t elastDCSTable);
__device__ void eHop(curandState *seed, Particle *electron_ptr, gFloat polar, gFloat azi);
__device__ void eTime(Particle *electron_ptr, gFloat ei);
__device__ void e2ndQ(Particle *electron_ptr, float edrop, Data *container, Particle *e2Queue, gFloat e2nd, int *e2nd_num, gFloat polar_e2nd, gFloat azi_e2nd, int *where, int *gEid);
__device__ void eRecord(Particle *electron_ptr, Data *container, float edrop, int *where, int ContainerSize);

__device__ void actChoice_proton(curandState *seed, cudaTextureObject_t protonCSTable, Particle *particle_ptr);
__device__ float sampleWIon(curandState *seed, Particle *particle_ptr);
__device__ float sampleWExc(curandState *seed, Particle *particle_ptr);
__device__ void eDrop_proton(curandState *seed, Particle *particle_ptr, float *e2nd, float *edrop);
__device__ void eAngles_proton(curandState *seed, Particle *particle_ptr, gFloat ei, gFloat e2nd, gFloat *polar, gFloat *azi, gFloat *polar_e2nd, gFloat *azi_e2nd);
__device__ void eHop_proton(curandState *seed, Particle *particle_ptr, gFloat polar, gFloat azi, float ei);

#endif