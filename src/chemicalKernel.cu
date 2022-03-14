#include "chemicalKernel.cuh"
#include "DNAKernel.cuh"

//gpu variables from chemical species
__device__ __constant__ float d_diffCoef_spec[MAXPARTYPE];
__device__ __constant__ float d_radii_spec[MAXPARTYPE];
__device__ __constant__ float d_maxDiffCoef_spec[1];

//gpu variables from reactions
__device__ int d_numReactant_React[MAXREACTION];
__device__ int d_indexReactant_React[MAXREACTION + 1];
__device__ int d_typeReactant_React[MAXREACTION * MAXNUMREACTANT4REACT];
__device__ int d_numNewPar_React[MAXREACTION];
__device__ int d_indexNewPar_React[MAXREACTION + 1];
__device__ int d_typeNewPar_React[MAXREACTION * MAXNUMNEWPAR4REACT];
__device__ int d_numReactant_Par[MAXPARTYPE];
__device__ int d_typeReactant_Par[MAXPARTYPE * MAXNUMREACTANT4PAR];
__device__ int d_subtypeReact_Par[MAXPARTYPE * MAXNUMREACTANT4PAR];	
__device__ float d_kobs_React[MAXREACTION];
__device__ float d_calc_radii_React[MAXREACTION * NUMDIFFTIMESTEPS];
__device__ float d_prob_React[MAXREACTION];

//gpu variables from radicals positions and types
texture<float,1,cudaReadModeElementType> posx_tex;
texture<float,1,cudaReadModeElementType> posy_tex;
texture<float,1,cudaReadModeElementType> posz_tex;
texture<unsigned char,1,cudaReadModeElementType> ptype_tex;

texture<float,1,cudaReadModeElementType> posx_d_tex;
texture<float,1,cudaReadModeElementType> posy_d_tex;
texture<float,1,cudaReadModeElementType> posz_d_tex;

__device__ int d_deltaidxBin_neig[27];
__device__ int d_numNewPar[1]; // the total number of new particles generated at current 
__device__ float d_deltaT[1];


__global__ void addOxygen(unsigned char* d_ptype, int* d_index, float* d_posx, float* d_posy,
    float* d_posz, float* d_ttime, int numCurPar, float minx, float maxx,
    float miny, float maxy, float minz, float maxz)
{
    int tid = blockIdx.x*blockDim.x+ threadIdx.x;
    curandState localState = cuseed[tid];
    float rando;
    if(tid<NUMOXYGEN)
    {
        d_ptype[tid+numCurPar] = 7;
        d_index[tid+numCurPar] = -1;
        d_ttime[tid+numCurPar] = 1;
        rando = curand_uniform(&localState);
        d_posz[tid+numCurPar] = minz - 250 + rando*(maxz-minz+500);
        rando = curand_uniform(&localState);
        d_posy[tid+numCurPar] = miny - 250 + rando*(maxy-miny+500);
        rando = curand_uniform(&localState);
        d_posx[tid+numCurPar] = minx - 250 + rando*(maxx-minx+500);
    }
    cuseed[tid] = localState;
}

__global__ void changePtype(unsigned char * ptype, int num,int targetValue)
{// change the ptype of first num radicals to targetValue
    int tid = blockIdx.x*blockDim.x+ threadIdx.x;
    if(tid<num)
        ptype[tid]=targetValue;
}

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

    extern __shared__ unsigned long sharedHash[];    // blockSize + 1 elements
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
    // Now use the sorted index to reorder the pos and vel data
        int sortedIndex = gridParticleIndex[tid];

        d_posx_s[tid] = tex1Dfetch(posx_tex, sortedIndex);
        d_posy_s[tid] = tex1Dfetch(posy_tex, sortedIndex);
        d_posz_s[tid] = tex1Dfetch(posz_tex, sortedIndex);
        d_ptype_s[tid] = tex1Dfetch(ptype_tex, sortedIndex);

        d_posx_sd[tid] = tex1Dfetch(posx_d_tex, sortedIndex);
        d_posy_sd[tid] = tex1Dfetch(posy_d_tex, sortedIndex);
        d_posz_sd[tid] = tex1Dfetch(posz_d_tex, sortedIndex);		
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
    // for(int tmpi=0;tmpi<numNewPar;tmpi++)
    // {
    //     if(d_typeNewPar_React[d_indexNewPar_React[reactType]+tmpi]==4)
    //         printf("tid %d react %d",tid, idx_par_neig);
    // }
    
    if(numNewPar == 0)
    {
        if(d_statusPar[tid] == 0 && d_statusPar[idx_par_neig] == 0)
        {
    
            d_statusPar[tid] = 255;
            d_statusPar[idx_par_neig] = 255;

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
            posx_new[tid] = pos_reactSite.x;
            posy_new[tid] = pos_reactSite.y;
            posz_new[tid] = pos_reactSite.z;
            ptype_new[tid] = d_typeNewPar_React[d_indexNewPar_React[reactType]];

            d_statusPar[tid] = 1;
            d_statusPar[idx_par_neig] = 255;

            ptype_new[idx_par_neig] = 255;

            return 1;
        }
        else if(numNewPar == 2 && d_statusPar[tid] == 0 && d_statusPar[idx_par_neig] == 0)
        {
            float costheta = curand_uniform(localState_pt);//revised by Youfang on Jan 11 2020
            float phi = 2*PI*curand_uniform(localState_pt);
            float ratio = curand_uniform(localState_pt);

            posx_new[tid] = pos_reactSite.x+ratio*calc_radii*sqrtf(1-costheta*costheta)*__cosf(phi);
            posy_new[tid] = pos_reactSite.y+ratio*calc_radii*sqrtf(1-costheta*costheta)*__sinf(phi);
            posz_new[tid] = pos_reactSite.z+ratio*calc_radii*costheta;
            ptype_new[tid] = d_typeNewPar_React[d_indexNewPar_React[reactType]];

            // the other generated new particle
            posx_new[idx_par_neig] = pos_reactSite.x+(1-ratio)*calc_radii*sqrtf(1-costheta*costheta)*__cosf(phi);
            posy_new[idx_par_neig] = pos_reactSite.y+(1-ratio)*calc_radii*sqrtf(1-costheta*costheta)*__cosf(phi);
            posz_new[idx_par_neig] = pos_reactSite.z+(1-ratio)*calc_radii*costheta;
            ptype_new[idx_par_neig] = d_typeNewPar_React[d_indexNewPar_React[reactType] + 1];

        
            d_statusPar[tid] = 1;
            d_statusPar[idx_par_neig] = 1;
                
            return 1;
        }
        else if(numNewPar == 3 && d_statusPar[tid] == 0 && d_statusPar[idx_par_neig] == 0)
        {
            float costheta = curand_uniform(localState_pt);
            float phi = 2*PI*curand_uniform(localState_pt);
            float ratio = curand_uniform(localState_pt);

            posx_new[tid] = pos_reactSite.x+ratio*calc_radii*sqrtf(1-costheta*costheta)*__cosf(phi);
            posy_new[tid] = pos_reactSite.y+ratio*calc_radii*sqrtf(1-costheta*costheta)*__sinf(phi);
            posz_new[tid] = pos_reactSite.z+ratio*calc_radii*costheta;
            ptype_new[tid] = d_typeNewPar_React[d_indexNewPar_React[reactType]];

            posx_new[idx_par_neig] = pos_reactSite.x+(1-ratio)*calc_radii*sqrtf(1-costheta*costheta)*__cosf(phi);
            posy_new[idx_par_neig] = pos_reactSite.y+(1-ratio)*calc_radii*sqrtf(1-costheta*costheta)*__cosf(phi);
            posz_new[idx_par_neig] = pos_reactSite.z+(1-ratio)*calc_radii*costheta;
            ptype_new[idx_par_neig] = d_typeNewPar_React[d_indexNewPar_React[reactType] + 1];

            posx_new[tid + numCurPar] = pos_reactSite.x;
            posy_new[tid + numCurPar] = pos_reactSite.y;
            posz_new[tid + numCurPar] = pos_reactSite.z;
            ptype_new[tid + numCurPar] = d_typeNewPar_React[d_indexNewPar_React[reactType] + 2];

            d_statusPar[tid] = 1;
            d_statusPar[idx_par_neig] = 1;

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

        float mintd = 1.0e6f;
        if(numPosReactant == 0) // this type of particle doesn't react with particles
        {
            d_mintd_Par[tid] = mintd;
            return 0; 
        }

        int kstart = d_accumParidxBin[idx_neig];
        int kend = d_accumParidxBin[idx_neig + 1];

        float3 pos_cur_neig;

        for(k=kstart; k<kend; k++)
        {
            if(k != tid && d_statusPar[k] == 0)
            {
                int ptype_neig = tex1Dfetch(ptype_tex, k); 
                                    
                pos_cur_neig.x = tex1Dfetch(posx_tex, k);
                pos_cur_neig.y = tex1Dfetch(posy_tex, k);	
                pos_cur_neig.z = tex1Dfetch(posz_tex, k);			   		   
                
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
                        
                        if(flag_react == 1) d_mintd_Par[tid] = 0.f;
                        
                        return flag_react;
                        }
                        else if(dis_cur_par_target_neig > calc_radii && d_statusPar[k] == 0 && d_statusPar[tid] == 0 && flag_prob == 1)
                        {
                        float temp = (dis_cur_par_target_neig - calc_radii)*(dis_cur_par_target_neig - calc_radii)/8.0f/(d_diffCoef_spec[ptype_target] + d_diffCoef_spec[ptype_neig] + 2.0f * sqrtf(d_diffCoef_spec[ptype_target]*d_diffCoef_spec[ptype_neig]));
                        
                        //printf("thread id = %d, temp = %f\n", tid, temp);
                        
                        if(mintd > temp) mintd = temp;
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

    if(tid < numCurPar && d_statusPar[tid] == 0)
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

    if(k != tid && d_statusPar[k] == 0)
    {
    int ptype_neig = tex1Dfetch(ptype_tex, k); 
                        
    pos_cur_neig.x = tex1Dfetch(posx_d_tex, k);
    pos_cur_neig.y = tex1Dfetch(posy_d_tex, k);	
    pos_cur_neig.z = tex1Dfetch(posz_d_tex, k);		   
    
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

        int ptype_target = tex1Dfetch(ptype_tex, tid);

        int numPosReactant = d_numReactant_Par[ptype_target];

        if(d_statusPar[tid] == 0 && numPosReactant > 0)
        { 
            int idx_in_nzbinidx = findBinIdxInNonZeroBinArray(binidx, d_nzBinidx, numNZBin);

            int numNeig = d_idxnzBin_numNeig[idx_in_nzbinidx];	

            int idx_neig = -1;

            for(int idx = 0; idx < numNeig; idx ++)
            {           
                //idx_neig = d_idxnzBin_neig[idx_in_nzbinidx * 27 + seq[idx]]; // return the index of the neighboring bin in the non-zero bin sequence. -1 denotes this neighboring bin has no particles inside
                idx_neig = d_idxnzBin_neig[idx_in_nzbinidx * 27 + idx];
                
                if(idx_neig > -1) 
                {
                    int flag_react = search4Reactant_beforeDiffusion(idx_neig, d_accumParidxBin, d_statusPar, posx_new, posy_new, posz_new, ptype_new, d_mintd_Par, &localState, numCurPar, idx_typeTimeStep);	
                    if(flag_react > 0) break;
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

        int ptype_target = tex1Dfetch(ptype_tex, tid);

        int numPosReactant = d_numReactant_Par[ptype_target];

        if(d_statusPar[tid] == 0 && numPosReactant > 0)
        { 
            int idx_in_nzbinidx = findBinIdxInNonZeroBinArray(binidx, d_nzBinidx, numNZBin);		    
            int numNeig = d_idxnzBin_numNeig[idx_in_nzbinidx];	
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

__global__ void makeOneJump4Diffusion(float *d_posx_d, float *d_posy_d, float *d_posz_d, int numCurPar) 
{//Radical diffuse one step
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

void ChemList::iniGPU()
{
    printf("\nStart GPU memory initialization\n");
	//gpu variables from chemical species	
	CUDA_CALL(cudaMemcpyToSymbol(d_diffCoef_spec, diffCoef_spec, sizeof(float)*numSpecType, 0, cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpyToSymbol(d_radii_spec, radii_spec, sizeof(float)*numSpecType, 0, cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpyToSymbol(d_maxDiffCoef_spec, &maxDiffCoef_spec, sizeof(float), 0, cudaMemcpyHostToDevice));

	//gpu variables from ReactionType class
	CUDA_CALL(cudaMemcpyToSymbol(d_numReactant_React, numReactant_React, sizeof(int)*numReact, 0, cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpyToSymbol(d_indexReactant_React, indexReactant_React, sizeof(int)*(numReact + 1), 0, cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpyToSymbol(d_typeReactant_React, typeReactant_React, sizeof(int)*indexReactant_React[numReact], 0, cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpyToSymbol(d_numNewPar_React, numNewPar_React, sizeof(int)*numReact, 0, cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpyToSymbol(d_indexNewPar_React, indexNewPar_React, sizeof(int)*(numReact + 1), 0, cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpyToSymbol(d_typeNewPar_React, typeNewPar_React, sizeof(int)*indexNewPar_React[numReact], 0, cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpyToSymbol(d_numReactant_Par, numReactant_Par, sizeof(float)*numSpecType, 0, cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpyToSymbol(d_typeReactant_Par, typeReactant_Par, sizeof(float)*numSpecType*MAXNUMREACTANT4PAR, 0, cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpyToSymbol(d_subtypeReact_Par, subtypeReact_Par, sizeof(float)*numSpecType*MAXNUMREACTANT4PAR, 0, cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpyToSymbol(d_kobs_React, kobs_React, sizeof(float)*numReact, 0, cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpyToSymbol(d_calc_radii_React, calc_radii_React, sizeof(float)*numReact * NUMDIFFTIMESTEPS, 0, cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpyToSymbol(d_prob_React, prob_React, sizeof(float)*numReact, 0, cudaMemcpyHostToDevice));
	
	int tempNumNewPar = 0;
	CUDA_CALL(cudaMemcpyToSymbol(d_numNewPar, &tempNumNewPar, sizeof(int), 0, cudaMemcpyHostToDevice));
}

void ChemList::copyDataToGPU()
{	
    printf("\nStart GPU memory initialization\n");
	//gpu variables from chemical species	
	CUDA_CALL(cudaMemcpyToSymbol(d_diffCoef_spec, diffCoef_spec, sizeof(float)*numSpecType, 0, cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpyToSymbol(d_radii_spec, radii_spec, sizeof(float)*numSpecType, 0, cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpyToSymbol(d_maxDiffCoef_spec, &maxDiffCoef_spec, sizeof(float), 0, cudaMemcpyHostToDevice));

	//gpu variables from ReactionType class
	CUDA_CALL(cudaMemcpyToSymbol(d_numReactant_React, numReactant_React, sizeof(int)*numReact, 0, cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpyToSymbol(d_indexReactant_React, indexReactant_React, sizeof(int)*(numReact + 1), 0, cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpyToSymbol(d_typeReactant_React, typeReactant_React, sizeof(int)*indexReactant_React[numReact], 0, cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpyToSymbol(d_numNewPar_React, numNewPar_React, sizeof(int)*numReact, 0, cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpyToSymbol(d_indexNewPar_React, indexNewPar_React, sizeof(int)*(numReact + 1), 0, cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpyToSymbol(d_typeNewPar_React, typeNewPar_React, sizeof(int)*indexNewPar_React[numReact], 0, cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpyToSymbol(d_numReactant_Par, numReactant_Par, sizeof(float)*numSpecType, 0, cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpyToSymbol(d_typeReactant_Par, typeReactant_Par, sizeof(float)*numSpecType*MAXNUMREACTANT4PAR, 0, cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpyToSymbol(d_subtypeReact_Par, subtypeReact_Par, sizeof(float)*numSpecType*MAXNUMREACTANT4PAR, 0, cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpyToSymbol(d_kobs_React, kobs_React, sizeof(float)*numReact, 0, cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpyToSymbol(d_calc_radii_React, calc_radii_React, sizeof(float)*numReact * NUMDIFFTIMESTEPS, 0, cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpyToSymbol(d_prob_React, prob_React, sizeof(float)*numReact, 0, cudaMemcpyHostToDevice));

	numCurPar = iniPar+NUMOXYGEN;
	maxPar = int(numCurPar * 2.1);
    //gpu variables for radicals positions and types
	CUDA_CALL(cudaMalloc((void **) &d_posx, sizeof(float)* maxPar));
	CUDA_CALL(cudaMemcpy(d_posx, posx, sizeof(float)*iniPar, cudaMemcpyHostToDevice));

	CUDA_CALL(cudaMalloc((void **) &d_posy, sizeof(float)*maxPar));
	CUDA_CALL(cudaMemcpy(d_posy, posy, sizeof(float)*iniPar, cudaMemcpyHostToDevice));

	CUDA_CALL(cudaMalloc((void **) &d_posz, sizeof(float)*maxPar));
	CUDA_CALL(cudaMemcpy(d_posz, posz, sizeof(float)*iniPar, cudaMemcpyHostToDevice));

	CUDA_CALL(cudaMalloc((void **) &d_ptype, sizeof(unsigned char)*maxPar));
	CUDA_CALL(cudaMemset(d_ptype, 255, sizeof(unsigned char) * maxPar));
	CUDA_CALL(cudaMemcpy(d_ptype, ptype, sizeof(unsigned char)*iniPar, cudaMemcpyHostToDevice));
	
	CUDA_CALL(cudaMalloc((void **) &d_index, sizeof(int)*maxPar));
	CUDA_CALL(cudaMemset(d_index, 0, sizeof(int) * maxPar));
	CUDA_CALL(cudaMemcpy(d_index, index, sizeof(int)*iniPar, cudaMemcpyHostToDevice));

	CUDA_CALL(cudaMalloc((void **) &d_ttime, sizeof(float)*maxPar));
	CUDA_CALL(cudaMemset(d_ttime, 0, sizeof(float) * maxPar));
	CUDA_CALL(cudaMemcpy(d_ttime, ttime, sizeof(float)*iniPar, cudaMemcpyHostToDevice));

    CUDA_CALL(cudaMalloc((void **) &d_statusPar, sizeof(unsigned char)*maxPar));

	CUDA_CALL(cudaMalloc((void **) &d_posx_s, sizeof(float)*maxPar)); // sorted array 
	CUDA_CALL(cudaMalloc((void **) &d_posy_s, sizeof(float)*maxPar));
	CUDA_CALL(cudaMalloc((void **) &d_posz_s, sizeof(float)*maxPar));
	CUDA_CALL(cudaMalloc((void **) &d_ptype_s, sizeof(unsigned int)*maxPar));
	
	CUDA_CALL(cudaMalloc((void **) &d_posx_d, sizeof(float)*maxPar)); // array after diffusion
	CUDA_CALL(cudaMalloc((void **) &d_posy_d, sizeof(float)*maxPar));
	CUDA_CALL(cudaMalloc((void **) &d_posz_d, sizeof(float)*maxPar));
	
	CUDA_CALL(cudaMalloc((void **) &d_posx_sd, sizeof(float)*maxPar)); // sorted arrat after diffusion
	CUDA_CALL(cudaMalloc((void **) &d_posy_sd, sizeof(float)*maxPar));
	CUDA_CALL(cudaMalloc((void **) &d_posz_sd, sizeof(float)*maxPar));
	
	CUDA_CALL(cudaMalloc((void **) &d_gridParticleHash, sizeof(unsigned long)*maxPar)); 
	CUDA_CALL(cudaMalloc((void **) &d_gridParticleIndex, sizeof(int)*maxPar));
	CUDA_CALL(cudaMalloc((void **) &d_accumParidxBin, sizeof(int)* (MAXNUMNZBIN + 1)));
	CUDA_CALL(cudaMalloc((void **) &d_nzBinidx, sizeof(unsigned long)* MAXNUMNZBIN));
	
	CUDA_CALL(cudaMalloc((void **) &d_idxnzBin_neig, sizeof(int)* MAXNUMNZBIN * 27));
    CUDA_CALL(cudaMalloc((void **) &d_idxnzBin_numNeig, sizeof(int)* MAXNUMNZBIN));
    	
	CUDA_CALL(cudaMalloc((void **) &d_mintd_Par, sizeof(float)*maxPar));
	
	h_mintd_Par_init = new float[maxPar];
	for(int i = 0; i< maxPar; i++)
	{
	   h_mintd_Par_init[i] = 1.0e6f;
	}
	
	int tempNumNewPar = 0;
	CUDA_CALL(cudaMemcpyToSymbol(d_numNewPar, &tempNumNewPar, sizeof(int), 0, cudaMemcpyHostToDevice));

    printf("Finish copying data to GPU for chemical stage\n\n");
}

void ChemList::run(DNAList ddl)
{	
    float max_posx, min_posx, max_posy, min_posy, max_posz, min_posz, mintd;
	
	float binSize, binSize_diffu;
    unsigned long numBinx, numBiny, numBinz,  numNZBin;//numBin,
	
	thrust::device_ptr<float> max_ptr;
	thrust::device_ptr<float> min_ptr;
			
	thrust::device_ptr<float> posx_dev_ptr;
	thrust::device_ptr<float> posy_dev_ptr;
	thrust::device_ptr<float> posz_dev_ptr;
	thrust::device_ptr<float> ttime_dev_ptr;
	thrust::device_ptr<int> index_dev_ptr;
	thrust::device_ptr<unsigned char> ptype_dev_ptr;
	
	thrust::device_ptr<unsigned long> uniBinidxPar_dev_ptr;
	
	thrust::device_ptr<float> posx_d_dev_ptr;
	thrust::device_ptr<float> posy_d_dev_ptr;
	thrust::device_ptr<float> posz_d_dev_ptr;
	
	thrust::device_ptr<float> mintd_dev_ptr;
	
	thrust::device_ptr<unsigned long> gridHash_dev_ptr;
	thrust::device_ptr<int> gridIndex_dev_ptr;
	thrust::device_ptr<int> numPar4Bin_dev_ptr;
	thrust::device_ptr<int> accumParIndex4Bin_dev_ptr;
	thrust::device_vector<unsigned long>::iterator result_unique_copy;
	thrust::device_vector<unsigned long>::iterator result_remove;

	thrust::device_vector<unsigned long> uniBinidxPar_dev_vec(30*numCurPar);
			
	typedef thrust::tuple<thrust::device_vector<unsigned char>::iterator, thrust::device_vector<float>::iterator, thrust::device_vector<float>::iterator, thrust::device_vector<float>::iterator,thrust::device_vector<float>::iterator,thrust::device_vector<int>::iterator> IteratorTuple;
        // define a zip iterator
		
	typedef thrust::zip_iterator<IteratorTuple> ZipIterator;
		
	ZipIterator zip_begin, zip_end, zip_new_end;
	
	int idx_iter = 0;
	
	int nblocks;
	int idx_typedeltaT;

	int idx_neig = 0;
	int numofextendbin=5;
	
	int totalIni=2*iniPar;
	recordposition = (float4*) malloc(sizeof(float4)*totalIni);
	memset(recordposition,0,sizeof(float4)*totalIni);
	CUDA_CALL(cudaMalloc((void**) &d_recordposition, sizeof(chemReact)*totalIni));
	printf("max_radi_react=%f,%f,%f,%f\n",max_calc_radii_React[0],max_calc_radii_React[1],max_calc_radii_React[2],max_calc_radii_React[3]);

    float process_time = document["chemicalTime"].GetFloat();
    float h_deltaT = 0.1;
/***********************************************************************************/	
	while(curTime < process_time) //curTime starts from 1
	{
		if(numCurPar==0) break;
		
		if(curTime < 10.0f) // choose default time interval
		    idx_typedeltaT = 0;
		else if(curTime < 100.0f)
		    idx_typedeltaT = 1;
		else if(curTime < 1000.0f)
		   idx_typedeltaT = 2;
		else if(curTime < 10000.0f)
		   idx_typedeltaT = 3;
		else
			idx_typedeltaT = 4;		
			
		h_deltaT = h_deltaT_adap[idx_typedeltaT];//delta T in current cycle
		
		binSize = 2 * max_calc_radii_React[idx_typedeltaT];
		
        if(idx_iter == 0)
        {
        	float maxsize = 2 * max_calc_radii_React[4];
        	numCurPar-=NUMOXYGEN;		
			posx_dev_ptr = thrust::device_pointer_cast(&d_posx[0]);
			max_ptr = thrust::max_element(posx_dev_ptr, posx_dev_ptr + numCurPar);
			max_posx=max_ptr[0]+numofextendbin*maxsize;
			min_ptr = thrust::min_element(posx_dev_ptr, posx_dev_ptr + numCurPar);
			min_posx=min_ptr[0]-numofextendbin*maxsize;
			
			posy_dev_ptr = thrust::device_pointer_cast(&d_posy[0]);
			max_ptr = thrust::max_element(posy_dev_ptr, posy_dev_ptr + numCurPar);
			max_posy=max_ptr[0]+numofextendbin*maxsize;
			min_ptr = thrust::min_element(posy_dev_ptr, posy_dev_ptr + numCurPar);
			min_posy=min_ptr[0]-numofextendbin*maxsize;
				
			posz_dev_ptr = thrust::device_pointer_cast(&d_posz[0]);
			max_ptr = thrust::max_element(posz_dev_ptr, posz_dev_ptr + numCurPar);
			max_posz=max_ptr[0]+numofextendbin*maxsize;
			min_ptr = thrust::min_element(posz_dev_ptr, posz_dev_ptr + numCurPar);
			min_posz=min_ptr[0]-numofextendbin*maxsize;			
			printf("max_posx = %f, min_posx = %f, max_posy = %f, min_posy = %f, max_posz = %f, min_posz = %f\n", max_posx, min_posx, max_posy, min_posy, max_posz, min_posz);			
			float V = (max_posz- min_posz+500)*(max_posx- min_posx+500)*(500+max_posy- min_posy)/1e9;
			printf("volume is %f um^3\n", V);
			if(NUMOXYGEN>0) addOxygen<<<1+(NUMOXYGEN-1)/512,512>>>(d_ptype, d_index, d_posx, d_posy, d_posz,
				d_ttime, numCurPar, min_posx, max_posx, min_posy, max_posy, min_posz, max_posz);
			numCurPar+= NUMOXYGEN;
		}
		//binSize=(binSize>(max_posx - min_posx)/200)?:(max_posx - min_posx)/200;
		numBinx = (max_posx - min_posx)/binSize + 1;
		numBiny = (max_posy - min_posy)/binSize + 1;
		numBinz = (max_posz - min_posz)/binSize + 1;
		//numBin = numBinx * numBiny * numBinz;
		
		nblocks = 1 + (numCurPar - 1)/NTHREAD_PERBLOCK_CHEM;
		assignBinidx4Par<<<nblocks,NTHREAD_PERBLOCK_CHEM>>>(d_gridParticleHash, d_gridParticleIndex, d_posx, d_posy, d_posz, min_posx, min_posy, min_posz, numBinx, numBiny, numBinz, binSize, numCurPar);		
		cudaDeviceSynchronize();		
				
		gridHash_dev_ptr = thrust::device_pointer_cast(&d_gridParticleHash[0]);
		gridIndex_dev_ptr = thrust::device_pointer_cast(&d_gridParticleIndex[0]);
		thrust::sort_by_key(gridHash_dev_ptr, gridHash_dev_ptr + numCurPar, gridIndex_dev_ptr);		
		
		result_unique_copy = thrust::unique_copy(gridHash_dev_ptr, gridHash_dev_ptr + numCurPar, uniBinidxPar_dev_vec.begin());		
		numNZBin = result_unique_copy - uniBinidxPar_dev_vec.begin();
		
		d_nzBinidx =  thrust::raw_pointer_cast(&uniBinidxPar_dev_vec[0]); 	

		nblocks = 1 + (numCurPar - 1)/NTHREAD_PERBLOCK_CHEM;
		FindParIdx4NonZeroBin<<<nblocks, NTHREAD_PERBLOCK_CHEM, (NTHREAD_PERBLOCK_CHEM+1)*sizeof(unsigned long)>>>(d_gridParticleHash, d_nzBinidx, d_accumParidxBin, numNZBin,numCurPar);
		cudaDeviceSynchronize();

		idx_neig = 0;		
		for(int iz = -1; iz < 2; iz ++)
	    {
	        for(int iy = -1; iy < 2; iy ++)
            {
		        for(int ix = -1; ix < 2; ix ++)
				{
				  h_deltaidxBin_neig[idx_neig] = iz * numBinx * numBiny + iy * numBinx + ix;//the linear index difference 				  
				  idx_neig++;
				}
			}
		}		
		CUDA_CALL(cudaMemcpyToSymbol(d_deltaidxBin_neig, h_deltaidxBin_neig, sizeof(int)*27, 0, cudaMemcpyHostToDevice));
		
		CUDA_CALL(cudaMemset(d_idxnzBin_numNeig, 0, sizeof(int) * numNZBin));		
		nblocks = 1 + (numNZBin * 27 - 1)/NTHREAD_PERBLOCK_CHEM;
		FindNeig4NonZeroBin<<<nblocks, NTHREAD_PERBLOCK_CHEM>>>(d_nzBinidx, d_idxnzBin_neig, d_idxnzBin_numNeig, numNZBin);
		cudaDeviceSynchronize();
	
		CUDA_CALL(cudaBindTexture(0, posx_tex, d_posx, sizeof(float) * numCurPar));
		CUDA_CALL(cudaBindTexture(0, posy_tex, d_posy, sizeof(float) * numCurPar));
		CUDA_CALL(cudaBindTexture(0, posz_tex, d_posz, sizeof(float) * numCurPar));
		CUDA_CALL(cudaBindTexture(0, ptype_tex, d_ptype, sizeof(unsigned char) * numCurPar));
		
		nblocks = 1 + (numCurPar - 1)/NTHREAD_PERBLOCK_CHEM;
		reorderData_beforeDiffusion<<<nblocks,NTHREAD_PERBLOCK_CHEM>>>(d_posx_s, d_posy_s, d_posz_s, d_ptype_s,d_gridParticleIndex, numCurPar);  //assign id after sorted                                      		
		cudaDeviceSynchronize();

		CUDA_CALL(cudaUnbindTexture(posx_tex));
		CUDA_CALL(cudaUnbindTexture(posy_tex));
		CUDA_CALL(cudaUnbindTexture(posz_tex));
		CUDA_CALL(cudaUnbindTexture(ptype_tex));
		
		CUDA_CALL(cudaMemset(d_statusPar, 255, sizeof(unsigned char) * maxPar)); // use 255 to mark the dead particle
		CUDA_CALL(cudaMemset(d_statusPar, 0, sizeof(unsigned char) * numCurPar));

		CUDA_CALL(cudaMemset(d_ptype, 255, sizeof(unsigned char) * maxPar)); // use 255 to mark the void entry in the new particle array
		CUDA_CALL(cudaMemcpy(d_ptype, d_ptype_s, sizeof(unsigned char) * numCurPar, cudaMemcpyDeviceToDevice));

		CUDA_CALL(cudaMemcpy(d_posx, d_posx_s, sizeof(float) * numCurPar, cudaMemcpyDeviceToDevice));
		CUDA_CALL(cudaMemcpy(d_posy, d_posy_s, sizeof(float) * numCurPar, cudaMemcpyDeviceToDevice));
		CUDA_CALL(cudaMemcpy(d_posz, d_posz_s, sizeof(float) * numCurPar, cudaMemcpyDeviceToDevice));
	
		CUDA_CALL(cudaMemcpy(d_mintd_Par, h_mintd_Par_init, sizeof(float)*numCurPar, cudaMemcpyHostToDevice));//min time, initilized to 1e6
		
		CUDA_CALL(cudaBindTexture(0, posx_tex, d_posx_s, sizeof(float) * numCurPar));
		CUDA_CALL(cudaBindTexture(0, posy_tex, d_posy_s, sizeof(float) * numCurPar));
		CUDA_CALL(cudaBindTexture(0, posz_tex, d_posz_s, sizeof(float) * numCurPar));
		CUDA_CALL(cudaBindTexture(0, ptype_tex, d_ptype_s, sizeof(unsigned char) * numCurPar));
/***********************************************************************************/	
        if(elapseReact>reactInterval || idx_iter==0)
        {
            CUDA_CALL(cudaMemset(d_recordposition, 0, sizeof(float4)*totalIni));
            nblocks = 1 + (numCurPar - 1)/NTHREAD_PERBLOCK_CHEM;
            reactDNA_beforeDiffusion<<<nblocks, NTHREAD_PERBLOCK_CHEM>>>(ddl.dev_chromatinIndex, ddl.dev_chromatinStart, ddl.dev_chromatinType,  ddl.dev_straightChrom,
                ddl.dev_bendChrom, ddl.dev_straightHistone, ddl.dev_bendHistone, d_statusPar, d_ptype, 
                                    d_mintd_Par, numCurPar, d_recordposition);
            cudaDeviceSynchronize();
            std::cout << "totalIni is "<<totalIni << std::endl;
            CUDA_CALL(cudaMemcpy(recordposition, d_recordposition, sizeof(float4)*totalIni, cudaMemcpyDeviceToHost));
            if(idx_iter!=0) saveResults();
            elapseReact = 0;
        }
        
/***********************************************************************************/
		nblocks = 1 + (numCurPar - 1)/NTHREAD_PERBLOCK_CHEM;
		react4TimeStep_beforeDiffusion<<<nblocks, NTHREAD_PERBLOCK_CHEM>>>(d_posx, d_posy, d_posz, d_ptype, d_gridParticleHash, d_idxnzBin_neig, d_idxnzBin_numNeig, d_nzBinidx, d_accumParidxBin, d_statusPar, d_mintd_Par, numBinx, numBiny, numBinz, numNZBin, numCurPar, idx_typedeltaT);
		cudaDeviceSynchronize();

		CUDA_CALL(cudaUnbindTexture(posx_tex));
		CUDA_CALL(cudaUnbindTexture(posy_tex));
		CUDA_CALL(cudaUnbindTexture(posz_tex));
		CUDA_CALL(cudaUnbindTexture(ptype_tex));			
		
		mintd_dev_ptr = thrust::device_pointer_cast(d_mintd_Par);
		min_ptr = thrust::min_element(mintd_dev_ptr, mintd_dev_ptr + numCurPar);
		mintd = min_ptr[0];		
//		printf("mintd = %f\n", mintd);

		/***************seems no need to clean extra particles here because if mintd==0, it will skip the diffusion part.
		if mintd>0, then no change of ptype before diffusion

		ptype_dev_ptr = thrust::device_pointer_cast(&d_ptype[0]);				
		zip_begin = thrust::make_zip_iterator(thrust::make_tuple(ptype_dev_ptr, posx_dev_ptr, posy_dev_ptr, posz_dev_ptr));
	    zip_end   = zip_begin + numCurPar * 2;  		
		zip_new_end = thrust::remove_if(zip_begin, zip_end, first_element_equal_255());

		numCurPar = zip_new_end - zip_begin;
		printf("current time is %f, after first step searching numCurPar = %d, the calculated mintd = %f\n", curTime, numCurPar, mintd);
		****************************************/

	    if(mintd > 0.0f) // some reactions occurs before diffusion, so no diffusion at this time step, delta t = 0
	    {
			if(mintd < h_deltaT || mintd >= 10000.0f)
			   mintd = h_deltaT;
			
			curTime += mintd;
			// printf("curTime = %f  mintd = %f\n", curTime, mintd);
			   
			cudaMemcpyToSymbol(d_deltaT, &mintd, sizeof(float), 0, cudaMemcpyHostToDevice);
			
			CUDA_CALL(cudaBindTexture(0, posx_tex, d_posx, sizeof(float) * numCurPar));//update d_posx in the above codes
			CUDA_CALL(cudaBindTexture(0, posy_tex, d_posy, sizeof(float) * numCurPar));
			CUDA_CALL(cudaBindTexture(0, posz_tex, d_posz, sizeof(float) * numCurPar));
			CUDA_CALL(cudaBindTexture(0, ptype_tex, d_ptype, sizeof(unsigned char) * numCurPar));
			
			nblocks = 1 + (numCurPar - 1)/NTHREAD_PERBLOCK_CHEM;
			makeOneJump4Diffusion<<<nblocks, NTHREAD_PERBLOCK_CHEM>>>(d_posx_d, d_posy_d, d_posz_d,numCurPar);
			cudaDeviceSynchronize();
		
			binSize_diffu = sqrt(6.0f * maxDiffCoef_spec * mintd); 
			
			if(binSize < binSize_diffu)
			{
			    binSize = binSize_diffu;	    
			}
			/*******************revised at Mar 3rd 2019. After diffusion, the minimun and maximum position should be updated***
			if not updated, the linear index can be wrong! (binidx_x can be larger than numBinx)*/
			posx_dev_ptr = thrust::device_pointer_cast(&d_posx_d[0]);
			max_ptr = thrust::max_element(posx_dev_ptr, posx_dev_ptr + numCurPar);
			max_posx=max_ptr[0]+numofextendbin*binSize;
			min_ptr = thrust::min_element(posx_dev_ptr, posx_dev_ptr + numCurPar);
			min_posx=min_ptr[0]-numofextendbin*binSize;
			
			posy_dev_ptr = thrust::device_pointer_cast(&d_posy_d[0]);
			max_ptr = thrust::max_element(posy_dev_ptr, posy_dev_ptr + numCurPar);
			max_posy=max_ptr[0]+numofextendbin*binSize;
			min_ptr = thrust::min_element(posy_dev_ptr, posy_dev_ptr + numCurPar);
			min_posy=min_ptr[0]-numofextendbin*binSize;
				
			posz_dev_ptr = thrust::device_pointer_cast(&d_posz_d[0]);
			max_ptr = thrust::max_element(posz_dev_ptr, posz_dev_ptr + numCurPar);
			max_posz=max_ptr[0]+numofextendbin*binSize;
			min_ptr = thrust::min_element(posz_dev_ptr, posz_dev_ptr + numCurPar);
			min_posz=min_ptr[0]-numofextendbin*binSize;						
			/*************************************************************************************/

	        numBinx = (max_posx - min_posx)/binSize + 1;
			numBiny = (max_posy - min_posy)/binSize + 1;
			numBinz = (max_posz - min_posz)/binSize + 1;
				
			nblocks = 1 + (numCurPar - 1)/NTHREAD_PERBLOCK_CHEM;
			assignBinidx4Par<<<nblocks,NTHREAD_PERBLOCK_CHEM>>>(d_gridParticleHash, d_gridParticleIndex, d_posx_d, d_posy_d, d_posz_d, min_posx, min_posy, min_posz, numBinx, numBiny, numBinz, binSize, numCurPar);			
			cudaDeviceSynchronize();	
		
			gridHash_dev_ptr = thrust::device_pointer_cast(&d_gridParticleHash[0]);
			gridIndex_dev_ptr = thrust::device_pointer_cast(&d_gridParticleIndex[0]);
			thrust::sort_by_key(gridHash_dev_ptr, gridHash_dev_ptr + numCurPar, gridIndex_dev_ptr);
		
			result_unique_copy = thrust::unique_copy(gridHash_dev_ptr, gridHash_dev_ptr + numCurPar, uniBinidxPar_dev_vec.begin());
			
			numNZBin = result_unique_copy - uniBinidxPar_dev_vec.begin();
			
			d_nzBinidx =  thrust::raw_pointer_cast(&uniBinidxPar_dev_vec[0]); 	
				
		    nblocks = 1 + (numCurPar - 1)/NTHREAD_PERBLOCK_CHEM;
			FindParIdx4NonZeroBin<<<nblocks, NTHREAD_PERBLOCK_CHEM, (NTHREAD_PERBLOCK_CHEM+1)*sizeof(unsigned long)>>>(d_gridParticleHash, d_nzBinidx, d_accumParidxBin, numNZBin,numCurPar);
			cudaDeviceSynchronize();
		
		
			idx_neig = 0;			
			for(int iz = -1; iz < 2; iz ++)
		    {
		        for(int iy = -1; iy < 2; iy ++)
	            {
			        for(int ix = -1; ix < 2; ix ++)
					{
					  h_deltaidxBin_neig[idx_neig] = iz * numBinx * numBiny + iy * numBinx + ix;
					  idx_neig++;
					}
				}
			}
		
			CUDA_CALL(cudaMemcpyToSymbol(d_deltaidxBin_neig, h_deltaidxBin_neig, sizeof(int)*27, 0, cudaMemcpyHostToDevice));
			
			cudaMemset(d_idxnzBin_numNeig, 0, sizeof(int) * numNZBin);
			
			nblocks = 1 + (numNZBin * 27 - 1)/NTHREAD_PERBLOCK_CHEM;
			FindNeig4NonZeroBin<<<nblocks, NTHREAD_PERBLOCK_CHEM>>>(d_nzBinidx, d_idxnzBin_neig, d_idxnzBin_numNeig, numNZBin);
			cudaDeviceSynchronize();

			cudaBindTexture(0, posx_d_tex, d_posx_d, sizeof(float) * numCurPar);
			cudaBindTexture(0, posy_d_tex, d_posy_d, sizeof(float) * numCurPar);
			cudaBindTexture(0, posz_d_tex, d_posz_d, sizeof(float) * numCurPar);
			
			nblocks = 1 + (numCurPar - 1)/NTHREAD_PERBLOCK_CHEM;
			reorderData_afterDiffusion<<<nblocks,NTHREAD_PERBLOCK_CHEM>>>(d_posx_s, d_posy_s, d_posz_s, d_ptype_s, d_posx_sd, d_posy_sd, d_posz_sd, d_gridParticleIndex, numCurPar);                                        
			cudaDeviceSynchronize();

			cudaUnbindTexture(posx_d_tex);
			cudaUnbindTexture(posy_d_tex);
			cudaUnbindTexture(posz_d_tex);
		    
			cudaUnbindTexture(posx_tex);
			cudaUnbindTexture(posy_tex);
			cudaUnbindTexture(posz_tex);
			cudaUnbindTexture(ptype_tex);

			cudaMemset(d_statusPar, 255, sizeof(unsigned char) * maxPar);
			cudaMemset(d_statusPar, 0, sizeof(unsigned char) * numCurPar);
			cudaMemset(d_ptype, 255, sizeof(unsigned char) * maxPar); // use 255 to mark the void entry in the new particle array					
			cudaMemcpy(d_ptype, d_ptype_s, sizeof(unsigned char) * numCurPar, cudaMemcpyDeviceToDevice);
			

			cudaMemcpy(d_posx, d_posx_sd, sizeof(float) * numCurPar, cudaMemcpyDeviceToDevice);
			cudaMemcpy(d_posy, d_posy_sd, sizeof(float) * numCurPar, cudaMemcpyDeviceToDevice);
			cudaMemcpy(d_posz, d_posz_sd, sizeof(float) * numCurPar, cudaMemcpyDeviceToDevice);	
			
		
		
			cudaBindTexture(0, posx_d_tex, d_posx_sd, sizeof(float) * numCurPar);
			cudaBindTexture(0, posy_d_tex, d_posy_sd, sizeof(float) * numCurPar);
			cudaBindTexture(0, posz_d_tex, d_posz_sd, sizeof(float) * numCurPar);
			
			cudaBindTexture(0, posx_tex, d_posx_s, sizeof(float) * numCurPar);
			cudaBindTexture(0, posy_tex, d_posy_s, sizeof(float) * numCurPar);
			cudaBindTexture(0, posz_tex, d_posz_s, sizeof(float) * numCurPar);
			cudaBindTexture(0, ptype_tex, d_ptype_s, sizeof(unsigned char) * numCurPar);

            elapseReact += mintd;
			if(elapseReact>reactInterval)
            {
                CUDA_CALL(cudaMemset(d_recordposition, 0, sizeof(float4)*totalIni)); //clear positions
                nblocks = 1 + (numCurPar - 1)/NTHREAD_PERBLOCK_CHEM;
                reactDNA_afterDiffusion<<<nblocks, NTHREAD_PERBLOCK_CHEM>>>(ddl.dev_chromatinIndex, ddl.dev_chromatinStart, ddl.dev_chromatinType,  ddl.dev_straightChrom,
                    ddl.dev_bendChrom, ddl.dev_straightHistone, ddl.dev_bendHistone, d_statusPar, mintd, d_ptype, numCurPar, d_recordposition); // should be modified to simply the function. Now it is a little bit redundant, by Youfang 05192021
                cudaDeviceSynchronize();
                CUDA_CALL(cudaMemcpy(recordposition, d_recordposition, sizeof(float4)*totalIni, cudaMemcpyDeviceToHost));
                saveResults();
            }		
			
			
			nblocks = 1 + (numCurPar - 1)/NTHREAD_PERBLOCK_CHEM;
			react4TimeStep_afterDiffusion<<<nblocks, NTHREAD_PERBLOCK_CHEM>>>(d_posx, d_posy, d_posz, d_ptype, d_gridParticleHash, d_idxnzBin_neig, d_idxnzBin_numNeig, d_nzBinidx, d_accumParidxBin, d_statusPar, numBinx, numBiny, numBinz, numNZBin, numCurPar, idx_typedeltaT);	
			cudaDeviceSynchronize();

			cudaUnbindTexture(posx_tex);
			cudaUnbindTexture(posy_tex);
			cudaUnbindTexture(posz_tex);
			cudaUnbindTexture(ptype_tex);
			
			cudaUnbindTexture(posx_d_tex);
			cudaUnbindTexture(posy_d_tex);
			cudaUnbindTexture(posz_d_tex);
	    }
		
	    posx_dev_ptr = thrust::device_pointer_cast(&d_posx[0]);			
		posy_dev_ptr = thrust::device_pointer_cast(&d_posy[0]);				
		posz_dev_ptr = thrust::device_pointer_cast(&d_posz[0]);		
		ptype_dev_ptr = thrust::device_pointer_cast(&d_ptype[0]);
		ttime_dev_ptr = thrust::device_pointer_cast(&d_ttime[0]);
		index_dev_ptr = thrust::device_pointer_cast(&d_index[0]);

		zip_begin = thrust::make_zip_iterator(thrust::make_tuple(ptype_dev_ptr, posx_dev_ptr, posy_dev_ptr, posz_dev_ptr, ttime_dev_ptr, index_dev_ptr));
	    zip_end   = zip_begin + numCurPar * 2;  		
		zip_new_end = thrust::remove_if(zip_begin, zip_end, chem_first_element_equal_255()); // remove dead entry, empty entry or water

		numCurPar = zip_new_end - zip_begin;		
		
		idx_iter++;

        elapseRecord += mintd;
        if(elapseRecord>recordInterval)
        {
            saveNvsTime();
            elapseRecord = 0;
        }

        if(idx_iter%100==0) printf("idx_iter = %d # of radicals = %d\n", idx_iter,numCurPar);   
    }
    free(recordposition);
    CUDA_CALL(cudaFree(d_recordposition));
    uniBinidxPar_dev_vec.clear();
}

__device__ int judge_par_before(int* dev_chromatinIndex,int* dev_chromatinStart,int* dev_chromatinType, CoorBasePair* dev_straightChrom,
    CoorBasePair* dev_bendChrom,float3* dev_straightHistone,float3* dev_bendHistone,float3 pos_cur_target,
    int3 index, int id, curandState* plocalState,float4* d_recordposition)
{
//judge whether the particle react with DNA, if yes, return flag 1;
int delta=index.x+index.y*NUCLEUS_DIM+index.z*NUCLEUS_DIM*NUCLEUS_DIM,minindex=-1;
float distance[3]={100},mindis=100;
CoorBasePair* chrom;
float3 *histone,newpos;
float chemprob=curand_uniform(plocalState);
int chromNum, histoneNum,tmp;
int ptype_target = tex1Dfetch(ptype_tex, id); // get the radical type

for(int i=0;i<27;i++)
{
int newindex = delta+neighborindex[i];
if(newindex<0 || newindex > NUCLEUS_DIM*NUCLEUS_DIM*NUCLEUS_DIM-1) continue;
int type = dev_chromatinType[newindex];
if(type==-1 || type==0) continue;
newpos = pos2local(type, pos_cur_target, newindex);
if(type<7)
{
if(newpos.x<(min1-SPACETOBODER) || newpos.y<(min2-SPACETOBODER) || newpos.z<(min3-SPACETOBODER) ||newpos.x>(max1+SPACETOBODER)
|| newpos.y>(max2+SPACETOBODER) || newpos.z>(max3+SPACETOBODER))
continue;
chrom=dev_straightChrom;
chromNum=STRAIGHT_BP_NUM;
histone=dev_straightHistone;
histoneNum=STRAIGHT_HISTONE_NUM;
}
else
{
if(newpos.x<(min1-SPACETOBODER) || newpos.y<(min2-SPACETOBODER) || newpos.z<(min3-SPACETOBODER) ||newpos.x>(max3+SPACETOBODER)
|| newpos.y>(max2+SPACETOBODER) || newpos.z>(max1+SPACETOBODER))
continue;
chrom=dev_bendChrom;
chromNum=BEND_BP_NUM;
histone=dev_bendHistone;
histoneNum=BEND_HISTONE_NUM;
}
for(int j=0;j<histoneNum;j++) // histone absorb all 
{
mindis = caldistance(newpos, histone[j])-RHISTONE;
if(mindis < 0) return 1;
}
if(ptype_target==1 || ptype_target==0) //only consider .OH and e- now
{	
for(int j=0;j<chromNum;j++)
{
// can take the size of base into consideration, distance should be distance-r;
mindis=100,minindex=-1;
distance[0] = caldistance(newpos, chrom[j].base)-RBASE;//exclude on-site situation
distance[1] = caldistance(newpos,chrom[j].left)-RSUGAR;
distance[2] = caldistance(newpos,chrom[j].right)-RSUGAR;
for(int iii=0;iii<3;iii++)
{
if(mindis>distance[iii])
{
mindis=distance[iii];
minindex=iii;
}
}
if(mindis<0)
{
if(ptype_target==1)
{
d_recordposition[id].x = pos_cur_target.x;
d_recordposition[id].y = pos_cur_target.y;
d_recordposition[id].z = pos_cur_target.z;
d_recordposition[id].w = 1;
}
return 1;
}

tmp = floorf(curand_uniform(plocalState)/0.25);//AGCT 1/4 probability

mindis=100,minindex=-1;
distance[0] -= d_rDNA[ptype_target*6+tmp];
distance[1] -= d_rDNA[ptype_target*6+4];
distance[2] -= d_rDNA[ptype_target*6+4];
for(int iii=0;iii<3;iii++)
{
if(mindis>distance[iii])
{
mindis=distance[iii];
minindex=iii;
}
}
if(mindis<0)
{
if(ptype_target==1)
{
d_recordposition[id].x = pos_cur_target.x;
d_recordposition[id].y = pos_cur_target.y;
d_recordposition[id].z = pos_cur_target.z;
d_recordposition[id].w = 1;
}

return 1;
}
}
}
}
return 0;
}
__global__ void reactDNA_beforeDiffusion(int* dev_chromatinIndex,int* dev_chromatinStart,int* dev_chromatinType, CoorBasePair* dev_straightChrom,
    CoorBasePair* dev_bendChrom,float3* dev_straightHistone,float3* dev_bendHistone,unsigned char* d_statusPar,unsigned char* d_type, 
    float* d_mintd_Par, int numCurPar, float4* d_recordposition)
{
int id = blockIdx.x*blockDim.x+ threadIdx.x;
curandState localState = cuseed[id];
if(id < numCurPar && d_statusPar[id] == 0)
{
float3 pos_cur_target;
pos_cur_target.x = tex1Dfetch(posx_tex, id);
pos_cur_target.y = tex1Dfetch(posy_tex, id);
pos_cur_target.z = tex1Dfetch(posz_tex, id);

int3 index;
index.x=floorf((pos_cur_target.x+UNITLENGTH*NUCLEUS_DIM/2)/UNITLENGTH);
index.y=floorf((pos_cur_target.y+UNITLENGTH*NUCLEUS_DIM/2)/UNITLENGTH);
index.z=floorf((pos_cur_target.z+UNITLENGTH*NUCLEUS_DIM/2)/UNITLENGTH);

int flag = judge_par_before(dev_chromatinIndex,dev_chromatinStart,dev_chromatinType,dev_straightChrom,dev_bendChrom, dev_straightHistone,dev_bendHistone,pos_cur_target,index, id, &localState, d_recordposition);
if(flag) 
{
d_statusPar[id]=255;
d_mintd_Par[id]=0; // although no procuts is considered now, obey the same rules to set it to 0
d_type[id]=255;
}
}
cuseed[id] = localState;
}

__device__ int judge_par_after(int* dev_chromatinIndex,int* dev_chromatinStart,int* dev_chromatinType, CoorBasePair* dev_straightChrom,
    CoorBasePair* dev_bendChrom,float3* dev_straightHistone,float3* dev_bendHistone,float3 pos_cur_target,
    float3 past, int3 index, int id, curandState* plocalState, float d_deltaT, float4* d_recordposition)
{
//judge whether the particle react with DNA, if yes, make its status be -1;
int delta=index.x+index.y*NUCLEUS_DIM+index.z*NUCLEUS_DIM*NUCLEUS_DIM,minindex=-1;
float distance[3]={100},mindis=100;
CoorBasePair* chrom;
float3 *histone;
float3 newpos,pastpos;
float dpre[3]={100}, prob_react;
float chemprob=curand_uniform(plocalState);
int chromNum, histoneNum,flag=0,tmp=0;
int ptype_target = tex1Dfetch(ptype_tex, id);
for(int i=0;i<27;i++)
{
int newindex = delta+neighborindex[i];
if(newindex<0 || newindex > NUCLEUS_DIM*NUCLEUS_DIM*NUCLEUS_DIM-1) continue;
int type = dev_chromatinType[newindex];
if(type==-1 || type==0) continue;
newpos = pos2local(type, pos_cur_target, newindex);

if(type<7)
{
if(newpos.x<(min1-SPACETOBODER) || newpos.y<(min2-SPACETOBODER) || newpos.z<(min3-SPACETOBODER) ||newpos.x>(max1+SPACETOBODER)
|| newpos.y>(max2+SPACETOBODER) || newpos.z>(max3+SPACETOBODER))
continue;
chrom=dev_straightChrom;
chromNum=STRAIGHT_BP_NUM;
histone=dev_straightHistone;
histoneNum=STRAIGHT_HISTONE_NUM;
}
else
{
if(newpos.x<(min1-SPACETOBODER) || newpos.y<(min2-SPACETOBODER) || newpos.z<(min3-SPACETOBODER) ||newpos.x>(max3+SPACETOBODER)
|| newpos.y>(max2+SPACETOBODER) || newpos.z>(max1+SPACETOBODER))
continue;
chrom=dev_bendChrom;
chromNum=BEND_BP_NUM;
histone=dev_bendHistone;
histoneNum=BEND_HISTONE_NUM;
}
pastpos=pos2local(type,past,newindex);

for(int j=0;j<histoneNum;j++)
{
mindis = caldistance(newpos, histone[j])-RHISTONE;
//printf("straight histone distance %F\n", distance);
if(mindis < 0) return 1;
else
{
dpre[0] = caldistance(pastpos, histone[j])-RHISTONE;
prob_react = expf(-1.0f*dpre[0]*mindis/d_diffCoef_spec[ptype_target]/d_deltaT);
if(curand_uniform(plocalState)<prob_react) return 1;
}
}	
if(ptype_target==1 || ptype_target==0) //only consider .OH and e- now
{	
for(int j=0;j<chromNum;j++)
{
mindis=100,minindex=-1;

distance[0] = caldistance(newpos, chrom[j].base)-RBASE;//exclude on-site situation
distance[1] = caldistance(newpos,chrom[j].left)-RSUGAR;
distance[2] = caldistance(newpos,chrom[j].right)-RSUGAR;
for(int iii=0;iii<3;iii++)
{
if(mindis>distance[iii])
{
mindis=distance[iii];
minindex=iii;
}
}
if(mindis<0)
{
if(ptype_target==1)
{
d_recordposition[id].x = pos_cur_target.x;
d_recordposition[id].y = pos_cur_target.y;
d_recordposition[id].z = pos_cur_target.z;
d_recordposition[id].w = 1;
}					
return 1;
}

mindis=100,minindex=-1;
tmp = floorf(curand_uniform(plocalState)/0.25);//AGCT 1/4 probability
dpre[0] = caldistance(pastpos, chrom[j].base)-RBASE- d_rDNA[ptype_target*6+tmp];
dpre[1] = caldistance(pastpos, chrom[j].left)-RSUGAR- d_rDNA[ptype_target*6+4];
dpre[2] = caldistance(pastpos, chrom[j].right)-RSUGAR- d_rDNA[ptype_target*6+4];
distance[0] -= d_rDNA[ptype_target*6+tmp];
distance[1] -= d_rDNA[ptype_target*6+4];
distance[2] -= d_rDNA[ptype_target*6+4];
mindis = 0;
for(int iii=0;iii<3;iii++)
{
prob_react = expf(-1.0f*dpre[iii]*distance[iii]/d_diffCoef_spec[ptype_target]/d_deltaT);
if(mindis<prob_react)
{
mindis=prob_react;//find maxium probability,do not be mislead by the name
minindex=iii;
}
}
if(curand_uniform(plocalState)<mindis) 
{
if(ptype_target==1)
{
d_recordposition[id].x = pos_cur_target.x;
d_recordposition[id].y = pos_cur_target.y;
d_recordposition[id].z = pos_cur_target.z;
d_recordposition[id].w = 1;
}
return 1;
}//*/
}
}	
}
return 0;
}
__global__ void reactDNA_afterDiffusion(int* dev_chromatinIndex,int* dev_chromatinStart,int* dev_chromatinType, CoorBasePair* dev_straightChrom,
    CoorBasePair* dev_bendChrom,float3* dev_straightHistone,float3* dev_bendHistone,unsigned char* d_statusPar, 
    float mintd, unsigned char* d_type, int numCurPar, float4* d_recordposition)
{
int id = blockIdx.x*blockDim.x+ threadIdx.x;
curandState localState = cuseed[id];
if(id < numCurPar && d_statusPar[id] == 0)
{
float3 pos_cur_target;
pos_cur_target.x = tex1Dfetch(posx_d_tex, id);
pos_cur_target.y = tex1Dfetch(posy_d_tex, id);
pos_cur_target.z = tex1Dfetch(posz_d_tex, id);

float3 pos_past_target;
pos_past_target.x = tex1Dfetch(posx_tex, id);
pos_past_target.y = tex1Dfetch(posy_tex, id);
pos_past_target.z = tex1Dfetch(posz_tex, id);

int3 index;
index.x=floorf((pos_cur_target.x+UNITLENGTH*NUCLEUS_DIM/2)/UNITLENGTH);
index.y=floorf((pos_cur_target.y+UNITLENGTH*NUCLEUS_DIM/2)/UNITLENGTH);
index.z=floorf((pos_cur_target.z+UNITLENGTH*NUCLEUS_DIM/2)/UNITLENGTH);

int flag=judge_par_after(dev_chromatinIndex,dev_chromatinStart,dev_chromatinType,  dev_straightChrom,dev_bendChrom,dev_straightHistone, dev_bendHistone,pos_cur_target,pos_past_target, index, id, &localState,mintd, d_recordposition);
if(flag) {d_statusPar[id]=11;d_type[id]=255;}
}
cuseed[id] = localState;
}