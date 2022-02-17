#ifndef CHEMICALKERNEL_CUH
#define CHEMICALKERNEL_CUH
#include "chemical.h"
#include "global.h"

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/functional.h>

struct chem_first_element_equal_255
{
  __host__ __device__
  bool operator()(const thrust::tuple<const unsigned char&, const float&, const float&, const float&, const float&, const int&> &t)
  {
      return thrust::get<0>(t) == 255;
  }
};

__global__ void addOxygen(unsigned char* d_ptype, int* d_index, float* d_posx, float* d_posy,
    float* d_posz, float* d_ttime, int numCurPar, float minx, float maxx,
    float miny, float maxy, float minz, float maxz);

__global__ void changePtype(unsigned char * ptype, int num,int targetValue);

__global__ void assignBinidx4Par(unsigned long   *d_gridParticleHash,  // output
                     int *d_gridParticleIndex,
                     float *d_posx,               // input: positions
                     float *d_posy,
                     float *d_posz,
                     const float min_posx,     // input: minimal positions
                     const float min_posy, 
                     const float min_posz, 
                     unsigned long numBinx, // number of bins in x dimension
                     unsigned long numBiny, 
                     unsigned long numBinz,
                     const float binSize,
                     int    numCurPar);
__device__ int findBinIdxInNonZeroBinArray(unsigned long binidx, unsigned long *d_nzBinidx, int numNZBin);
__global__ void FindParIdx4NonZeroBin(unsigned long *d_gridParticleHash,
                           unsigned long *d_nzBinidx,                           
                           int *d_accumParidxBin, 
                           int numNZBin,
                           int numCurPar);
__global__ void FindNeig4NonZeroBin(unsigned long *d_nzBinidx, int *d_idxnzBin_neig, int *d_idxnzBin_numNeig, int numNZBin);

__global__ void reorderData_beforeDiffusion(float *d_posx_s,        // output: sorted positions
                                    float *d_posy_s,        // output: sorted positions
                                    float *d_posz_s,        // output: sorted positions
                                    unsigned char *d_ptype_s, // output: sorted positions			
                                    int   *gridParticleIndex,// input: sorted particle indices
                                    int    numCurPar);

__global__ void reorderData_afterDiffusion(float *d_posx_s,        // output: sorted positions
                               float *d_posy_s,        // output: sorted positions
                               float *d_posz_s,        // output: sorted positions
                               unsigned char *d_ptype_s, // output: sorted positions
                               float *d_posx_sd, 
                               float *d_posy_sd, 
                               float *d_posz_sd, 
                               int   *gridParticleIndex,// input: sorted particle indices
                               int    numCurPar);
__device__ int generateNewPar(int reactType,
                  float calc_radii, 
                  float dis_par_target_neig, 
                  int ptype_target, 
                  int ptype_neig, 
                  int idx_par_neig, 		 
                  float3 *pos_target, 
                  float3 *pos_neig, 					
                  unsigned char *d_statusPar, 
                  float *posx_new, 
                  float *posy_new, 
                  float *posz_new, 
                  unsigned char *ptype_new,
                  curandState *localState_pt,
                  int numCurPar);
__device__ int search4Reactant_beforeDiffusion(int idx_neig, int *d_accumParidxBin, unsigned char *d_statusPar, float *posx_new, float *posy_new, float *posz_new, unsigned char *ptype_new, float *d_mintd_Par, curandState *localState_pt, int numCurPar, int idx_typeTimeStep);
__device__ int search4Reactant_afterDiffusion(int idx_neig, int *d_accumParidxBin, unsigned char *d_statusPar, float *posx_new, float *posy_new, float *posz_new, unsigned char *ptype_new, curandState *localState_pt, int numCurPar, int idx_typeTimeStep);
__global__ void react4TimeStep_beforeDiffusion(float *posx_new, 
                                   float *posy_new, 
                                   float *posz_new, 
                                   unsigned char *ptype_new,
                                   unsigned long *gridParticleHash,
                                   int *d_idxnzBin_neig,
                                   int *d_idxnzBin_numNeig,
                                   unsigned long *d_nzBinidx,
                                   int *d_accumParidxBin,
                                   unsigned char *d_statusPar,
                                   float *d_mintd_Par,
                                   unsigned long numBinx, // number of bins in x dimension
                                   unsigned long numBiny, 
                                   unsigned long numBinz,
                                   int numNZBin,
                                   int numCurPar,
                                   int idx_typeTimeStep);
__global__ void react4TimeStep_afterDiffusion(float *posx_new, 
                                   float *posy_new, 
                                   float *posz_new, 
                                   unsigned char *ptype_new,
                                   unsigned long *gridParticleHash,
                                   int *d_idxnzBin_neig,
                                   int *d_idxnzBin_numNeig,
                                   unsigned long *d_nzBinidx,
                                   int *d_accumParidxBin,
                                   unsigned char *d_statusPar,
                                   unsigned long numBinx, // number of bins in x dimension
                                   unsigned long numBiny, 
                                   unsigned long numBinz,
                                   int numNZBin,
                                   int numCurPar,
                                   int idx_typeTimeStep);
__global__ void makeOneJump4Diffusion(float *d_posx_d, float *d_posy_d, float *d_posz_d, int numCurPar);

__device__ int judge_par_before(int* dev_chromatinIndex,int* dev_chromatinStart,int* dev_chromatinType, CoorBasePair* dev_straightChrom,
								CoorBasePair* dev_bendChrom,float3* dev_straightHistone,float3* dev_bendHistone,float3 pos_cur_target,
								int3 index, int id, curandState* plocalState,float4* d_recordposition);
__global__ void reactDNA_beforeDiffusion(int* dev_chromatinIndex,int* dev_chromatinStart,int* dev_chromatinType, CoorBasePair* dev_straightChrom,
								CoorBasePair* dev_bendChrom,float3* dev_straightHistone,float3* dev_bendHistone,unsigned char* d_statusPar,unsigned char* d_type, 
								float* d_mintd_Par, int numCurPar, float4* d_recordposition);
__global__ void reactDNA_afterDiffusion(int* dev_chromatinIndex,int* dev_chromatinStart,int* dev_chromatinType, CoorBasePair* dev_straightChrom,
    CoorBasePair* dev_bendChrom,float3* dev_straightHistone,float3* dev_bendHistone,unsigned char* d_statusPar, 
    float mintd, unsigned char* d_type, int numCurPar, float4* d_recordposition);
__device__ int judge_par_after(int* dev_chromatinIndex,int* dev_chromatinStart,int* dev_chromatinType, CoorBasePair* dev_straightChrom,
    CoorBasePair* dev_bendChrom,float3* dev_straightHistone,float3* dev_bendHistone,float3 pos_cur_target,
    float3 past, int3 index, int id, curandState* plocalState, float d_deltaT, float4* d_recordposition);               
#endif