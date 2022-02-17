#ifndef DNAKERNEL_CUH
#define DNAKERNEL_CUH

#include "DNAList.h"
#include "global.h"

extern __constant__  int neighborindex[27];
extern __constant__ float min1, min2, min3, max1, max2, max3;

extern  __constant__  float d_rDNA[72];


__device__ float caldistance(float3 pos1, float3 pos2);
__device__ float3 pos2local(int type, float3 pos, int index);
__global__ void phySearch(int num, Edeposit* d_edrop, int* dev_chromatinIndex,int* dev_chromatinStart,int* dev_chromatinType, CoorBasePair* dev_straightChrom,
								CoorBasePair* dev_bendChrom,float3* dev_straightHistone,float3* dev_bendHistone, combinePhysics* d_recorde);
__global__ void chemSearch(int num, Edeposit* d_edrop, int* dev_chromatinIndex,int* dev_chromatinStart,int* dev_chromatinType, CoorBasePair* dev_straightChrom,
								CoorBasePair* dev_bendChrom,float3* dev_straightHistone,float3* dev_bendHistone, combinePhysics* d_recorde);

#endif