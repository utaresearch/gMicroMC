#ifndef __MICROMC_KERNELS_H__
#define __MICROMC_KERNELS_H__

#include <stdio.h>


/*__global__ void assignBinidx4Par(float *d_posx, float *d_posy, float *d_posz, int *d_binidxPar, int *d_accumParidxBin, const float min_posx, const float min_posy, const float min_posz, const int numBinx, const int numBiny, const int numBinz, const float dt, const float binSize, const int numCurPar)
{
  const int tid = blockIdx.x*blockDim.x+ threadIdx.x;
  if(tid < numCurPar)
  {
    int binidx_x = int((d_posx[tid] - min_posx)/binSize);
    int binidx_y = int((d_posy[tid] - min_posy)/binSize);	
	int binidx_z = int((d_posz[tid] - min_posz)/binSize);	
	d_binidxPar[tid] = numBinx * numBiny * binidx_z + numBinx * binidx_y + binidx_x;
	
	if(tid == 564)
	printf("tid = %d, d_posx[tid] = %f, d_posy[tid] = %f, d_posz[tid] = %f, binidx_x = %d, binidx_y = %d, binidx_z = %d, d_binidxPar = %d\n", tid, d_posx[tid], d_posy[tid], d_posz[tid], binidx_x, binidx_y, binidx_z, d_binidxPar[tid]); 
    atomicAdd(&d_accumParidxBin[d_binidxPar[tid]], 1);
  }  
}*/

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
                                 int    numCurPar)
{
    const int tid = blockIdx.x*blockDim.x+ threadIdx.x;
    
	if(tid < numCurPar)
    {
        unsigned long binidx_x = ((d_posx[tid] - min_posx)/binSize);
        unsigned long binidx_y = ((d_posy[tid] - min_posy)/binSize);	
	    unsigned long binidx_z = ((d_posz[tid] - min_posz)/binSize);
		
	    d_gridParticleHash[tid] = numBinx * numBiny * binidx_z + numBinx * binidx_y + binidx_x;	
		d_gridParticleIndex[tid] = tid;
		
	}	    
}


__device__ int findBinIdxInNonZeroBinArray(unsigned long binidx, unsigned long *d_nzBinidx, int numNZBin)
{
    const int tid = blockIdx.x*blockDim.x+ threadIdx.x;
	 
    int numSearch = int(__log2f(numNZBin))+1;
    
	int istart = 0;
	int iend = numNZBin - 1;
	unsigned long nzbinidx;
	int imiddle;
	int iSearch;
	
	int flag = -1;
	
    for(iSearch = 0; iSearch < numSearch; iSearch++)
    {
	  imiddle = istart + (iend - istart)/2; 
	  
	  nzbinidx = d_nzBinidx[imiddle];
	  
	  if(binidx == nzbinidx)
	  {
		flag = imiddle; 
        return flag;		
	  }
	  else if(binidx < nzbinidx)
	  { 
	    iend = imiddle - 1;
	  }
	  else
	  {
	    istart = imiddle + 1;
	  }
    }
 	  
    return flag;
}


__global__ void FindParIdx4NonZeroBin(unsigned long *d_gridParticleHash,
                                       unsigned long *d_nzBinidx,                           
                                       int *d_accumParidxBin, 
									   int numNZBin,
									   int numCurPar)
{
  const int tid = blockIdx.x*blockDim.x+ threadIdx.x;
  
  __shared__ unsigned long sharedHash[NTHREAD_PER_BLOCK_PAR+1];    // blockSize + 1 elements
  unsigned long binidx;
  
   if(tid < numCurPar)
   {
    binidx = d_gridParticleHash[tid];
	
	sharedHash[threadIdx.x+1] = binidx;

    if (tid > 0 && threadIdx.x == 0)
    {
        // first thread in block must load neighbor particle hash
        sharedHash[0] = d_gridParticleHash[tid-1];
    }
	
   }
   __syncthreads();
   
    if(tid < numCurPar)
    {
	    if(tid == 0 || binidx != sharedHash[threadIdx.x])
        {           
		
	        int flag = findBinIdxInNonZeroBinArray(binidx, d_nzBinidx, numNZBin);
			if(flag > 0)
			d_accumParidxBin[flag] = tid;        
        } 
        
        if(tid == numCurPar - 1)
        d_accumParidxBin[numNZBin] = numCurPar;   		
		
		if(tid == 0)
		d_accumParidxBin[0] = 0;   	
    }
}

/*__global__ void FindParIdx4NonZeroBin(unsigned long *d_gridParticleHash,
                                       unsigned long *d_nzBinidx,                           
                                       int *d_accumParidxBin, 
									   int numNZBin,
									   int numCurPar)
{
  const int tid = blockIdx.x*blockDim.x+ threadIdx.x;
  
 // __shared__ unsigned long sharedHash[NTHREAD_PER_BLOCK_PAR+1];    // blockSize + 1 elements
  unsigned long binidx;
     
    if(tid < numCurPar)
    {
	    binidx = d_gridParticleHash[tid];
		
	    if(tid == 0 || binidx != d_gridParticleHash[tid-1])
        {           
		
	        int flag = findBinIdxInNonZeroBinArray(binidx, d_nzBinidx, numNZBin);
			if(flag > 0)
			d_accumParidxBin[flag] = tid;        
        } 
        
        if(tid == numCurPar - 1)
        d_accumParidxBin[numNZBin] = numCurPar;   		
		
		if(tid == 0)
		d_accumParidxBin[0] = 0;   	
    }
}*/

__global__ void FindNeig4NonZeroBin(unsigned long *d_nzBinidx, int *d_idxnzBin_neig, int *d_idxnzBin_numNeig, int numNZBin)
{
    const int tid = blockIdx.x*blockDim.x+ threadIdx.x;
	int istart;
	int result;
	
	if(tid < numNZBin * 27)
	{
	  d_idxnzBin_neig[tid] = -1;
	  
	  int idxNZBin = tid/27;
	  int idx = tid - idxNZBin * 27;
	  
	  int d_idx_neig = d_deltaidxBin_neig[idx];
	  
	  if(d_idx_neig == 0)
	  {
	    istart = atomicAdd(&d_idxnzBin_numNeig[idxNZBin], 1);
	    d_idxnzBin_neig[idxNZBin * 27 + istart] = idxNZBin;	   
	  }
	  else
	  {
	   result = findBinIdxInNonZeroBinArray(d_nzBinidx[idxNZBin] + d_idx_neig, d_nzBinidx, numNZBin); 
	   if(result >-1)
	   {
	     istart = atomicAdd(&d_idxnzBin_numNeig[idxNZBin], 1);
	     d_idxnzBin_neig[idxNZBin * 27 + istart] = result; 
	   }
      }	   
	}
}

__global__ void reorderData_beforeDiffusion(float *d_posx_s,        // output: sorted positions
							                float *d_posy_s,        // output: sorted positions
							                float *d_posz_s,        // output: sorted positions
							                unsigned char *d_ptype_s, // output: sorted positions			
                                            int   *gridParticleIndex,// input: sorted particle indices
                                            int    numCurPar)
{
   
    const int tid = blockIdx.x*blockDim.x+ threadIdx.x;

    if (tid < numCurPar)
    {
        // If this particle has a different cell index to the previous
        // particle then it must be the first particle in the cell,
        // so store the index of this particle in the cell.
        // As it isn't the first particle, it must also be the cell end of
        // the previous particle's cell
     	
        // Now use the sorted index to reorder the pos and vel data
        int sortedIndex = gridParticleIndex[tid];
        
		d_posx_s[tid] = tex1Dfetch(posx_tex, sortedIndex);
		d_posy_s[tid] = tex1Dfetch(posy_tex, sortedIndex);
		d_posz_s[tid] = tex1Dfetch(posz_tex, sortedIndex);
		d_ptype_s[tid] = tex1Dfetch(ptype_tex, sortedIndex);
		
    }
}

__global__ void reorderData_afterDiffusion(float *d_posx_s,        // output: sorted positions
							               float *d_posy_s,        // output: sorted positions
										   float *d_posz_s,        // output: sorted positions
										   unsigned char *d_ptype_s, // output: sorted positions
										   float *d_posx_sd, 
										   float *d_posy_sd, 
										   float *d_posz_sd, 
                            			   int   *gridParticleIndex,// input: sorted particle indices
                            			   int    numCurPar)
{
   
    const int tid = blockIdx.x*blockDim.x+ threadIdx.x;

    if (tid < numCurPar)
    {
        // If this particle has a different cell index to the previous
        // particle then it must be the first particle in the cell,
        // so store the index of this particle in the cell.
        // As it isn't the first particle, it must also be the cell end of
        // the previous particle's cell
     	
        // Now use the sorted index to reorder the pos and vel data
        int sortedIndex = gridParticleIndex[tid];
        
		d_posx_s[tid] = tex1Dfetch(posx_tex, sortedIndex);
		d_posy_s[tid] = tex1Dfetch(posy_tex, sortedIndex);
		d_posz_s[tid] = tex1Dfetch(posz_tex, sortedIndex);
		d_ptype_s[tid] = tex1Dfetch(ptype_tex, sortedIndex);
		
		d_posx_sd[tid] = tex1Dfetch(posx_d_tex, sortedIndex);
		d_posy_sd[tid] = tex1Dfetch(posy_d_tex, sortedIndex);
		d_posz_sd[tid] = tex1Dfetch(posz_d_tex, sortedIndex);		
		/*if(tid == 0)
		{
		  printf("sortedIndex = %d, d_posx_s = %f, d_posy_s = %f, d_posz_s = %f, d_ptype_s = %u\n", sortedIndex, d_posx_s[tid], d_posy_s[tid], d_posz_s[tid], d_ptype_s[tid]);
		  printf("d_posx_sd = %f, d_posy_sd = %f, d_posz_sd = %f\n", d_posx_sd[tid], d_posy_sd[tid], d_posz_sd[tid]);
		}*/	
    }
}

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
							  int numCurPar)
{
    const int tid = blockIdx.x*blockDim.x+ threadIdx.x;
		
	int numNewPar = d_numNewPar_React[reactType];
	
	if(numNewPar == 0)
	{
	  
	  if(d_statusPar[tid] == 0 && d_statusPar[idx_par_neig] == 0)
	  {
	    // printf("target id = %d, neig id = %d, target pos = %f, %f, %f, neigh pos = %f, %f, %f, dis_par_target_neig = %f, calc_radii = %f, reactType = %d, ptype_target = %d, ptype_neig = %d, numNewPar = %d\n", tid, idx_par_neig, pos_target->x, pos_target->y, pos_target->z, pos_neig->x, pos_neig->y, pos_neig->z, dis_par_target_neig, calc_radii, reactType, ptype_target, ptype_neig, numNewPar);
		 
		 d_statusPar[tid] = -1;
		 d_statusPar[idx_par_neig] = -1;
		 
		 ptype_new[tid] = 255; 
		 ptype_new[idx_par_neig] = 255; 
		 		 		 
		 return 1;
	  }  
	}
	else
	{
	    float3 pos_reactSite;
		
		float ratio1 = sqrtf(d_diffCoef_spec[ptype_neig])/(sqrtf(d_diffCoef_spec[ptype_target]) + sqrtf(d_diffCoef_spec[ptype_neig]));
	    float ratio2 = sqrtf(d_diffCoef_spec[ptype_target])/(sqrtf(d_diffCoef_spec[ptype_target]) + sqrtf(d_diffCoef_spec[ptype_neig]));
	   
		pos_reactSite.x = ratio1 * pos_target->x + ratio2 * pos_neig->x;
		pos_reactSite.y = ratio1 * pos_target->y + ratio2 * pos_neig->y;
		pos_reactSite.z = ratio1 * pos_target->z + ratio2 * pos_neig->z;
	   
	    if(numNewPar == 1 && d_statusPar[tid] == 0 && d_statusPar[idx_par_neig] == 0)
		{
		   posx_new[tid] = ratio1 * pos_target->x + ratio2 * pos_neig->x;
	       posy_new[tid] = ratio1 * pos_target->y + ratio2 * pos_neig->y;
	       posz_new[tid] = ratio1 * pos_target->z + ratio2 * pos_neig->z;
	       ptype_new[tid] = d_typeNewPar_React[d_indexNewPar_React[reactType]];
		   
		  //printf("target id = %d, neig id = %d, target pos = %f, %f, %f, neigh pos = %f, %f, %f,dis_par_target_neig = %f, calc_radii = %f, reactType = %d, ptype_target = %d, ptype_neig = %d, numNewPar = %d, ptype_new = %d\n", tid, idx_par_neig, pos_target->x, pos_target->y, pos_target->z, pos_neig->x, pos_neig->y, pos_neig->z,dis_par_target_neig, calc_radii, reactType, ptype_target, ptype_neig, numNewPar, d_typeNewPar_React[d_indexNewPar_React[reactType]]);
		   
		   d_statusPar[tid] = -1;
		   d_statusPar[idx_par_neig] = -1;
		   
		   ptype_new[idx_par_neig] = 255;
		   
		   return 1;
		}
		else if(numNewPar == 2 && d_statusPar[tid] == 0 && d_statusPar[idx_par_neig] == 0)
		{		   
		   posx_new[tid] = pos_reactSite.x;
	       posy_new[tid] = pos_reactSite.y;
	       posz_new[tid] = pos_reactSite.z;
		   ptype_new[tid] = d_typeNewPar_React[d_indexNewPar_React[reactType]];
		   
		   // the other generated new particle
	       posx_new[idx_par_neig] = pos_reactSite.x;
	       posy_new[idx_par_neig] = pos_reactSite.y;
	       posz_new[idx_par_neig] = pos_reactSite.z;
		   ptype_new[idx_par_neig] = d_typeNewPar_React[d_indexNewPar_React[reactType] + 1];
		   
		   //printf("target id = %d, neig id = %d, target pos = %f, %f, %f, neigh pos = %f, %f, %f, dis_par_target_neig = %f, calc_radii = %f, reactType = %d, ptype_target = %d, ptype_neig = %d, numNewPar = %d, ptype_new1 = %d, ptype_new2 = %d\n", tid, idx_par_neig, pos_target->x, pos_target->y, pos_target->z, pos_neig->x, pos_neig->y, pos_neig->z,dis_par_target_neig, calc_radii, reactType, ptype_target, ptype_neig, numNewPar, d_typeNewPar_React[d_indexNewPar_React[reactType]], d_typeNewPar_React[d_indexNewPar_React[reactType]+1]);
		   
		   d_statusPar[tid] = -1;
		   d_statusPar[idx_par_neig] = -1;
		   	   
		   return 1;
		}
		else if(numNewPar == 3 && d_statusPar[tid] == 0 && d_statusPar[idx_par_neig] == 0)
		{
		   posx_new[tid] = pos_reactSite.x;
	       posy_new[tid] = pos_reactSite.y;
	       posz_new[tid] = pos_reactSite.z;
		   ptype_new[tid] = d_typeNewPar_React[d_indexNewPar_React[reactType]];
		   
		   posx_new[idx_par_neig] = pos_reactSite.x;
	       posy_new[idx_par_neig] = pos_reactSite.y;
	       posz_new[idx_par_neig] = pos_reactSite.z;
		   ptype_new[idx_par_neig] = d_typeNewPar_React[d_indexNewPar_React[reactType] + 1];
		   
		   posx_new[tid + numCurPar] = pos_reactSite.x;
	       posy_new[tid + numCurPar] = pos_reactSite.y;
	       posz_new[tid + numCurPar] = pos_reactSite.z;
		   ptype_new[tid + numCurPar] = d_typeNewPar_React[d_indexNewPar_React[reactType] + 2];
		   
		   d_statusPar[tid] = -1;
		   d_statusPar[idx_par_neig] = -1;
		   
		  //printf("target id = %d, neig id = %d, target pos = %f, %f, %f, neigh pos = %f, %f, %f, dis_par_target_neig = %f, calc_radii = %f, reactType = %d, ptype_target = %d, ptype_neig = %d, numNewPar = %d, ptype_new1 = %d, ptype_new2 = %d, ptype_new3 = %d\n", tid, idx_par_neig, pos_target->x, pos_target->y, pos_target->z, pos_neig->x, pos_neig->y, pos_neig->z,dis_par_target_neig, calc_radii, reactType, ptype_target, ptype_neig, numNewPar, d_typeNewPar_React[d_indexNewPar_React[reactType]], d_typeNewPar_React[d_indexNewPar_React[reactType]+1], d_typeNewPar_React[d_indexNewPar_React[reactType]+2]);
		   
		   return 1;
		}
	}
	
	return 0;	

}
__device__ int search4Reactant_beforeDiffusion(int idx_neig, int *d_accumParidxBin, unsigned char *d_statusPar, float *posx_new, float *posy_new, float *posz_new, unsigned char *ptype_new, float *d_mintd_Par, curandState *localState_pt, int numCurPar, int idx_typeTimeStep)		
//-------------------------------------------------------------------------------------------------------
// the function to search for potential reactant in the current searching bin for the target particle;
//  if one reactant is find, then reaction occurs and return 1 as a reaction flag
//  if no potential reactant is found, return 0.
//--------------------------------------------------------------------------------------------------------
{
    const int tid = blockIdx.x*blockDim.x+ threadIdx.x;
	
	if(tid < numCurPar && d_statusPar[tid] == 0)
	{
	float3 pos_cur_target;
	pos_cur_target.x = tex1Dfetch(posx_tex, tid);
	pos_cur_target.y = tex1Dfetch(posy_tex, tid);
	pos_cur_target.z = tex1Dfetch(posz_tex, tid);
	
	int ptype_target = tex1Dfetch(ptype_tex, tid);
		
	int k;
	
	int numPosReactant = d_numReactant_Par[ptype_target];
	
	if(numPosReactant == 0) // this type of particle doesn't react with particles
	{
	  return 0; 
	}
	
	int kstart = d_accumParidxBin[idx_neig];
	int kend = d_accumParidxBin[idx_neig + 1];
	
	float3 pos_cur_neig;
	 	
	float mintd = 1.0e6f;
	
	for(k=kstart; k<kend; k++)
	{
	   
		
	    if(k != tid)
		{
		   int ptype_neig = tex1Dfetch(ptype_tex, k); 
				  		   
		   pos_cur_neig.x = tex1Dfetch(posx_tex, k);
           pos_cur_neig.y = tex1Dfetch(posy_tex, k);	
		   pos_cur_neig.z = tex1Dfetch(posz_tex, k);
		   		   
		   if (d_statusPar[tid] != 0)
		   {
		      return 0;
		   }
		   
		   if (d_statusPar[k] != 0)
		   {
		      continue;
		   }   
		   for(int i = 0; i < numPosReactant; i ++)
		    {		        
			 			
				if(ptype_neig == d_typeReactant_Par[MAXNUMREACTANT4PAR * ptype_target + i])
			    {
			        float dis_cur_par_target_neig = sqrtf((pos_cur_target.x - pos_cur_neig.x) * (pos_cur_target.x - pos_cur_neig.x) + (pos_cur_target.y - pos_cur_neig.y) * (pos_cur_target.y - pos_cur_neig.y) + (pos_cur_target.z - pos_cur_neig.z) * (pos_cur_target.z - pos_cur_neig.z));
					
					int reactType = d_subtypeReact_Par[MAXNUMREACTANT4PAR * ptype_target + i];
																                
					float calc_radii = d_calc_radii_React[reactType * NUMDIFFTIMESTEPS + idx_typeTimeStep];
					
					float prob_react = d_prob_React[reactType];
					
					int flag_prob = 1;
					
					//printf("tid = %d, k = %d, ptype_target = %d, ptype_neig = %d, dis_cur_par_target_neig = %f, calc_radii = %f\n", tid, k, ptype_target, ptype_neig, dis_cur_par_target_neig, calc_radii);
								
					if(prob_react < 1.0f && curand_uniform(localState_pt) > prob_react)
					{
					    flag_prob = 0;
					}
					   
					if(dis_cur_par_target_neig <= calc_radii && d_statusPar[k] == 0 && d_statusPar[tid] == 0 && flag_prob == 1)
					{   
					   int flag_react = generateNewPar(reactType, calc_radii, dis_cur_par_target_neig, ptype_target, ptype_neig, k, &pos_cur_target, &pos_cur_neig, d_statusPar, posx_new, posy_new, posz_new, ptype_new, localState_pt, numCurPar);					   
					  
					   if(flag_react == 1) 
					   d_mintd_Par[tid] = 0.f;
					   
					   return flag_react;
					}
					else if(dis_cur_par_target_neig > calc_radii && d_statusPar[k] == 0 && d_statusPar[tid] == 0 && flag_prob == 1)
					{
					   float temp = (dis_cur_par_target_neig - calc_radii)*(dis_cur_par_target_neig - calc_radii)/8.0f/(d_diffCoef_spec[ptype_target] + d_diffCoef_spec[ptype_neig] + 2.0f * sqrtf(d_diffCoef_spec[ptype_target]*d_diffCoef_spec[ptype_neig]));
					  
					   //printf("thread id = %d, temp = %f\n", tid, temp);
					  
 					  if(mintd > temp) 
					   mintd = temp;
					}					
					break;
			    }
				
		    }         		   
		}
	}
	
	  d_mintd_Par[tid] = mintd;
	}
	
	return 0;
	
}

__device__ int search4Reactant_afterDiffusion(int idx_neig, int *d_accumParidxBin, unsigned char *d_statusPar, float *posx_new, float *posy_new, float *posz_new, unsigned char *ptype_new, curandState *localState_pt, int numCurPar, int idx_typeTimeStep)		
//-------------------------------------------------------------------------------------------------------
// the function to search for potential reactant in the current searching bin for the target particle;
//  if one reactant is find, then reaction occurs and return 1 as a reaction flag
//  if no potential reactant is found, return 0.
//--------------------------------------------------------------------------------------------------------
{
    const int tid = blockIdx.x*blockDim.x+ threadIdx.x;
	
	if(tid < numCurPar)
	{
	float3 pos_cur_target;
	pos_cur_target.x = tex1Dfetch(posx_d_tex, tid);
	pos_cur_target.y = tex1Dfetch(posy_d_tex, tid);
	pos_cur_target.z = tex1Dfetch(posz_d_tex, tid);
	
	int ptype_target = tex1Dfetch(ptype_tex, tid);
	int k;
	
	int numPosReactant = d_numReactant_Par[ptype_target];
	
	if(numPosReactant == 0) // this type of particle doesn't react with particles
	{
	  return 0; 
	}
	
	int kstart = d_accumParidxBin[idx_neig];
	int kend = d_accumParidxBin[idx_neig + 1];
	
	float3 pos_cur_neig;
	
	for(k=kstart; k<kend; k++)
	{
	 
	    if(k != tid)
		{
		   int ptype_neig = tex1Dfetch(ptype_tex, k); 
				  		   
		   pos_cur_neig.x = tex1Dfetch(posx_d_tex, k);
           pos_cur_neig.y = tex1Dfetch(posy_d_tex, k);	
		   pos_cur_neig.z = tex1Dfetch(posz_d_tex, k);		   
		   
		   if (d_statusPar[tid] != 0)
		   {
		      return 0;
		   }
		   
		   if (d_statusPar[k] != 0)
		   {
		      continue;
		   }   
		   for(int i = 0; i < numPosReactant; i ++)
		    {		        
			 			
				if(ptype_neig == d_typeReactant_Par[MAXNUMREACTANT4PAR * ptype_target + i])
			    {
			        float dis_cur_par_target_neig = sqrtf((pos_cur_target.x - pos_cur_neig.x) * (pos_cur_target.x - pos_cur_neig.x) + (pos_cur_target.y - pos_cur_neig.y) * (pos_cur_target.y - pos_cur_neig.y) + (pos_cur_target.z - pos_cur_neig.z) * (pos_cur_target.z - pos_cur_neig.z));
					
					int reactType = d_subtypeReact_Par[MAXNUMREACTANT4PAR * ptype_target + i];
															             				
					float calc_radii = d_calc_radii_React[reactType * NUMDIFFTIMESTEPS + idx_typeTimeStep];
					
					float prob_react = d_prob_React[reactType];
					
					int flag_prob = 1;
													
					if(prob_react < 1.0f && curand_uniform(localState_pt) > prob_react)
					{
					    flag_prob = 0;
					}
					   
					if(dis_cur_par_target_neig <= calc_radii && d_statusPar[k] == 0 && d_statusPar[tid] == 0 && flag_prob == 1)
					{   
					   int flag_react = generateNewPar(reactType, calc_radii, dis_cur_par_target_neig, ptype_target, ptype_neig, k, &pos_cur_target, &pos_cur_neig, d_statusPar, posx_new, posy_new, posz_new, ptype_new, localState_pt, numCurPar);					   					 
					   return flag_react;
					}
					else if(dis_cur_par_target_neig > calc_radii && d_statusPar[k] == 0 && d_statusPar[tid] == 0 && flag_prob == 1)
					{
					   // brownian bridge
					   
                       float3 pos_past_target, pos_past_neig;
					   
					   pos_past_target.x = tex1Dfetch(posx_tex, tid);
                       pos_past_target.y = tex1Dfetch(posy_tex, tid);	
		               pos_past_target.z = tex1Dfetch(posz_tex, tid);	
					   
					   pos_past_neig.x = tex1Dfetch(posx_tex, k);
                       pos_past_neig.y = tex1Dfetch(posy_tex, k);	
		               pos_past_neig.z = tex1Dfetch(posz_tex, k);		
		   
					   float dis_past_par_target_neig = sqrtf((pos_past_target.x - pos_past_neig.x) * (pos_past_target.x - pos_past_neig.x) + (pos_past_target.y - pos_past_neig.y) * (pos_past_target.y - pos_past_neig.y) + (pos_past_target.z - pos_past_neig.z) * (pos_past_target.z - pos_past_neig.z));
					   
					   prob_react = expf(-1.0f*(dis_past_par_target_neig - calc_radii) * (dis_cur_par_target_neig - calc_radii)/(d_diffCoef_spec[ptype_target] + d_diffCoef_spec[ptype_neig])/d_deltaT[0]);
					   
					   float temprand = curand_uniform(localState_pt);
					  
					   if(temprand < prob_react)
					   {
					      //printf("tid = %d, dis_cur_par_target_neig = %f, dis_past_par_target_neig = %f, prob_react = %f, temprand = %f, calc_radii = %f, d_diffCoef_spec[ptype_target] = %f, d_diffCoef_spec[ptype_neig] = %f, d_deltaT = %f\n", tid, dis_cur_par_target_neig, dis_past_par_target_neig, prob_react, temprand, calc_radii, d_diffCoef_spec[ptype_target], d_diffCoef_spec[ptype_neig], d_deltaT[0]);					 
						  int flag_react = generateNewPar(reactType, calc_radii, dis_cur_par_target_neig, ptype_target, ptype_neig, k, &pos_cur_target, &pos_cur_neig, d_statusPar, posx_new, posy_new, posz_new, ptype_new, localState_pt, numCurPar);					   					   
					      return flag_react;
					   }
					}					
					break;
			    }				
		    }         		   
		}
	}
	
	}
	
	return 0;
	
}

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
								               int idx_typeTimeStep)
{
   const int tid = blockIdx.x*blockDim.x+ threadIdx.x;
   
   if(tid < numCurPar)
    {
        unsigned long binidx = gridParticleHash[tid];  
		  
	    curandState localState = cuseed[tid];
	  
	    if(d_statusPar[tid] == 0)
	    { 
	        int idx_in_nzbinidx = findBinIdxInNonZeroBinArray(binidx, d_nzBinidx, numNZBin);
		    
	        int numNeig = d_idxnzBin_numNeig[idx_in_nzbinidx];	
			
			/*int seq[27]; 
			
			int tnum_pm = 1; // total number of different permutations
			
			for(int i = 0; i<numNeig; i++)
            {
		        seq[i] = i;
			    tnum_pm = tnum_pm * (i+1);
            }
                
			if(numNeig >1)
			{
                
 				int radnum = curand_uniform(&localState) * tnum_pm;
				int times_pm = 0;
				
				int flag_pm = mypermute_onecomb(seq, 0, numNeig-1, radnum, &times_pm);				
			}*/
			
			int idx_neig = -1;
			
	        for(int idx = 0; idx < numNeig; idx ++)
		    {
		      
				//idx_neig = d_idxnzBin_neig[idx_in_nzbinidx * 27 + seq[idx]]; // return the index of the neighboring bin in the non-zero bin sequence. -1 denotes this neighboring bin has no particles inside
		      	idx_neig = d_idxnzBin_neig[idx_in_nzbinidx * 27 + idx];
				
		        if(idx_neig > -1) 
				{
		            int flag_react = search4Reactant_beforeDiffusion(idx_neig, d_accumParidxBin, d_statusPar, posx_new, posy_new, posz_new, ptype_new, d_mintd_Par, &localState, numCurPar, idx_typeTimeStep);	
					if(flag_react > 0)
					break;
				}				
            }				
	    }
		
		cuseed[tid] = localState;
    }
}

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
								               int idx_typeTimeStep)
{
   const int tid = blockIdx.x*blockDim.x+ threadIdx.x;
   
   if(tid < numCurPar)
    {
        unsigned long binidx = gridParticleHash[tid];  
		  
	    curandState localState = cuseed[tid];
	  
	    if(d_statusPar[tid] == 0)
	    { 
	        int idx_in_nzbinidx = findBinIdxInNonZeroBinArray(binidx, d_nzBinidx, numNZBin);
		    
			int numNeig = d_idxnzBin_numNeig[idx_in_nzbinidx];	
			
			/*int seq[27];
			int tnum_pm = 1; // total number of different permutations
			
			for(int i = 0; i<numNeig; i++)
            {
		        seq[i] = i;
			    tnum_pm = tnum_pm * (i+1);
            }
				
			if(numNeig >1)
			{
                  
 				int radnum = curand_uniform(&localState) * tnum_pm;
				int times_pm = 0;
				
				int flag_pm = mypermute_onecomb(seq, 0, numNeig-1, radnum, &times_pm);
				
				//if(numNeig == 6)
				//printf("tid = %d, flag_pm = %d, seq[0] = %d, seq[1] = %d, seq[2] = %d, seq[3] = %d, seq[4] = %d, seq[5] = %d, radnum = %d\n", tid, flag_pm, seq[0], seq[1], seq[2], seq[3], seq[4], seq[5], radnum);
			}*/
			
			int idx_neig = -1;
			
	        for(int idx = 0; idx < numNeig; idx ++)
		    {
		       
				//idx_neig = d_idxnzBin_neig[idx_in_nzbinidx * 27 + seq[idx]]; // return the index of the neighboring bin in the non-zero bin sequence. -1 denotes this neighboring bin has no particles inside
		        idx_neig = d_idxnzBin_neig[idx_in_nzbinidx * 27 + idx];
				
		        if(idx_neig > -1) 
				{
		            int flag_react = search4Reactant_afterDiffusion(idx_neig, d_accumParidxBin, d_statusPar, posx_new, posy_new, posz_new, ptype_new, &localState, numCurPar, idx_typeTimeStep);	
					if(flag_react > 0)
					break;
				}				
            }				
	    }
		
		cuseed[tid] = localState;
    }
}

__global__ void makeOneJump4Diffusion(float *d_posx_d, 
                                      float *d_posy_d, 
								      float *d_posz_d,                               
								      int numCurPar)
{
    const int tid = blockIdx.x*blockDim.x+ threadIdx.x;
	
	if(tid < numCurPar)
	{
	    curandState localState = cuseed[tid]; 
		
		unsigned char ptype_target = tex1Dfetch(ptype_tex, tid);
	
		float zeta1 = -2.0f * log(curand_uniform(&localState));
        float zeta2 = 2.0f * PI * curand_uniform(&localState);
		
		float std_dis_diffusion = __fsqrt_rn(2.0f * d_diffCoef_spec[ptype_target] * d_deltaT[0]);
		
		d_posx_d[tid] = tex1Dfetch(posx_tex, tid) + std_dis_diffusion * __fsqrt_rn(zeta1)* __cosf(zeta2);
		d_posy_d[tid] = tex1Dfetch(posy_tex, tid) + std_dis_diffusion * __fsqrt_rn(zeta1)* __sinf(zeta2);
		
		zeta1 = -2.0f * log(curand_uniform(&localState));
		zeta2 = 2.0f * PI * curand_uniform(&localState);
		d_posz_d[tid] = tex1Dfetch(posz_tex, tid) + std_dis_diffusion * __fsqrt_rn(zeta1)* __cosf(zeta2);
		
	    cuseed[tid] = localState;
	}
 
}

#endif
