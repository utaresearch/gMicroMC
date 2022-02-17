#ifndef PRECHEMICAL_CUH
#define PRECHEMICAL_CUH

#include "global.h"
#include <thrust/device_vector.h>

struct first_element_equal_255
{
  __host__ __device__
  bool operator()(const thrust::tuple<const int&, const float&, const float&, const float&, const int&, const float&> &t)
  {
      return thrust::get<0>(t) == 255;
  }
};

__global__ void thermalisation_subexelectrons(float *d_posx, // x position of the particles (input and output)
                                              float *d_posy,
											  float *d_posz,
											  float *d_ene, // initial energies of the initial particles (input only, no use as of May 2021)
											  int *d_ptype, // species type for products of prechemical stage, 255 for empty or produced water (output)
											  int *d_statetype, //the statetype of the initial particles (255 for died particles)
											  int *d_wiid_elec); // the index of the ionized water molecule for potential recombination

__global__ void dissociation_ionizedwater(float *d_posx,
                                          float *d_posy,
										  float *d_posz,
										  int *d_ptype,
										  int *d_statetype);

__global__ void dissociation_excitedwater_a1b1(float *d_posx,
                                               float *d_posy,
										       float *d_posz,
										       int *d_ptype,
										       int *d_statetype);

__global__ void dissociation_excitedwater_b1a1(float *d_posx,
                                               float *d_posy,
										       float *d_posz,
										       int *d_ptype,
										       int *d_statetype);

__global__ void dissociation_excitedwater_rd(float *d_posx,
                                             float *d_posy,
										     float *d_posz,
										     int *d_ptype,
										     int *d_statetype);

__global__ void dissociation_dissociativewater(float *d_posx,
                                               float *d_posy,
										       float *d_posz,
										       int *d_ptype,
										       int *d_statetype);

__device__ void displace_twoproducts_noholehoping(float *d_posx, 
                                                  float *d_posy, 
												  float *d_posz,
												  curandState *localState_pt,
												  int btype, //branch type
												  int pid, // the current particle id
												  int pid_site);// the id of the particle considerred to be the original site (for recombination) 

__device__ void displace_threeproducts_noholehoping(float *d_posx, 
                                                  float *d_posy, 
												  float *d_posz, 
												  curandState *localState_pt,
												  int btype, //branch type
												  int pid, // the current particle id
												  int pid_site); // the id of the particle considerred to be the original site (for recombination) 

__device__ void displace_twoproducts_holehoping(float *d_posx, 
                                                float *d_posy, 
												float *d_posz, 
												curandState *localState_pt,
												int btype, //branch type
												int pid); // the current particle id        

__device__ void sampleThermalDistance(int pid, curandState *localState_pt, float *ndisx, float *ndisy, float *ndisz, float idx_ebin);

__device__ void displace_twoproducts_oneelec_holehoping(float *d_posx, 
                                                        float *d_posy, 
												        float *d_posz, 
												        curandState *localState_pt,
												        int btype, //branch type
												        int pid); // the current particle id

__device__ void getNormalizedDis_Sample3DGuassian(curandState *localState_pt, float *ndisx, float *ndisy, float *ndisz) ;

__device__ void getDirection_SampleOnSphereSurface(curandState *localState_pt, float *ndisx, float *ndisy, float *ndisz);                                                        	                                                                                                                                                                                                                                 	                                                                                                                                       


#endif