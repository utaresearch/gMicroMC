#include "physicsKernel.cuh"
#include "globalKernel.cuh"
#include "physicsList.h"

__constant__ float eEcutoff, pECutoff; // extern variable, defined by the user
__constant__ float3 boundaryCenter, boundarySize, ROICenter, ROISize;
__constant__ int boundaryShape, ROIShape;
// global varian in this file
texture<float,1,cudaReadModeElementType> eneProb_tex;
__constant__  float mecc = 5.11e5, Mpcc = 9.38e8, Ryd = 13.6;
__constant__  float I[5] = {539.7, 32.2, 18.44, 14.7, 12.6};
__constant__ float parameters[18] = {1.25,0.5,1,1,3,1.1,1.3,1,0,1.02,82,0.45,-0.8,0.38,1.07,14.6,0.6,0.04};
__constant__ float alpha[2] = {0.66,0.64};

__constant__ float foi[3] = {0.0187,0.0157,0.7843};
__constant__ float alphai[3] = {3, 1, 0.6};
__constant__ float Woi[3] = {8.4, 10.1, 21.3};

__global__ void setParticles(int num, int ptype, float A, float RMAX, Particle* d_eQueue)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = id;
    float X, Y, Z, r,cost,phi, e, prob;
    curandState localstate = cuseed[tid];
    float nBins = (tex1D(eneProb_tex, 0.5)-1);
    float emin = tex1D(eneProb_tex, 1.5);
    float deltaE = tex1D(eneProb_tex, 2.5);
    while(id<num)
    {
        r= RMAX*cbrtf(curand_uniform(&localstate));
        cost=-1+2*curand_uniform(&localstate);
        phi=2*PI*curand_uniform(&localstate);
        d_eQueue[id].x = r*sqrtf(1-cost*cost)*__cosf(phi);
        d_eQueue[id].y = r*sqrtf(1-cost*cost)*__sinf(phi);
        d_eQueue[id].z = r*cost;
        
        cost=-1+2*curand_uniform(&localstate);
        phi=2*PI*curand_uniform(&localstate);       
        d_eQueue[id].ux = sqrtf(1-cost*cost)*__cosf(phi);
        d_eQueue[id].uy = sqrtf(1-cost*cost)*__sinf(phi);
        d_eQueue[id].uz = cost;

        /*X = -RMAX/2.0 + RMAX*(curand_uniform(&localstate)); 
        Y = -RMAX/2.0 + RMAX*(curand_uniform(&localstate)); 
        Z = -RMAX/2.0;
        d_eQueue[id].x = X; 
        d_eQueue[id].y = Y; 
        d_eQueue[id].z = Z; 
        d_eQueue[id].ux = 0; 
        d_eQueue[id].uy = 0; 
        d_eQueue[id].uz = 1; */

        do{
            e = nBins*curand_uniform(&localstate);
            prob = tex1D(eneProb_tex, 3.5+e);
        }while(prob<curand_uniform(&localstate));
        d_eQueue[id].e = emin+deltaE*e;

        d_eQueue[id].h2oState = 99;
        d_eQueue[id].dead = 0;
        d_eQueue[id].path = 0;     
        d_eQueue[id].elape_time = 0.0;     
        d_eQueue[id].id = id;     
        d_eQueue[id].parentId = -1;
        d_eQueue[id].ptype = ptype;
        d_eQueue[id].A = A;

        id += blockDim.x*gridDim.x;
    } 
    cuseed[tid]=localstate;  
}
void sampleSource(int num, int ptype, float A, float R, float *h_eneprob, Particle *h_particles)
{
    int nBins = h_eneprob[0];
    cudaArray *d_eneProb;
    CUDA_CALL(cudaMallocArray(&d_eneProb, &eneProb_tex.channelDesc, nBins+3, 1));
    CUDA_CALL(cudaMemcpyToArray(d_eneProb, 0, 0, h_eneprob, sizeof(float)*(nBins+3), cudaMemcpyHostToDevice));
    eneProb_tex.filterMode = cudaFilterModeLinear;
    CUDA_CALL(cudaBindTextureToArray(eneProb_tex, d_eneProb));

    Particle* d_particles;
    cudaMalloc((void**)&d_particles,sizeof(Particle)*num);
    setParticles<<<60,256>>>(num, ptype, A, R, d_particles);
    cudaDeviceSynchronize();
    cudaMemcpy(h_particles,d_particles,sizeof(Particle)*num,cudaMemcpyDeviceToHost);

    cudaFree(d_particles);
    CUDA_CALL( cudaUnbindTexture(eneProb_tex));
    CUDA_CALL( cudaFreeArray(d_eneProb));
}

void sortEofElectron(Particle *dev_e2Queue, int sim_num)
{
    printf("sorting by thrust\n");
    thrust::device_ptr<Particle> dev_ptr(dev_e2Queue);
    thrust::sort(thrust::device, dev_ptr, dev_ptr+sim_num, sortEDescend<Particle>());
}

void PhysicsList::transportParticles()
// transport particles according to physiciType. -1 electron, 1 proton 
// record simulation time into totalTime
{
    cudaEvent_t start, stop;
    float milliseconds = 0.0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);        
    cudaEventRecord(start, 0);
    if(physicsPType == 1)
        pTransport<<<120, 512>>>(sim_num, ContainerSize, dev_pQueue, dev_e2Queue, dev_container, dev_where, dev_second_num, dev_gEid,
                                    texObj_protonTable, MaxN);
    else
        eTransport<<<120, 512>>>(sim_num, ContainerSize, dev_eQueue, dev_e2Queue, dev_container, dev_where, dev_second_num, dev_gEid,
            texObj_DACSTable, texObj_BindE_array, texObj_ieeCSTable, texObj_elastDCSTable, MaxN);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaDeviceSynchronize();

    cudaEventElapsedTime(&milliseconds, start, stop);

    totalTime += milliseconds;
}

__device__ float fetchData(cudaTextureObject_t CSTable, int i)
{
    return tex2D<gFloat>(CSTable, i/11+0.5, i%11+0.5);
}
__global__ void printTex(cudaTextureObject_t CSTable)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<1)
    {
        for(int j =0;j<800;j++)
        {
            if(j%11==0) printf("\n");
            float tmp = fetchData(CSTable, j);
            printf("%f ",tmp);
        }           
        printf("random number, eEcutoff %f %f\n", curand_uniform(&cuseed[i]), eEcutoff);
    }
}

void runPrint(cudaTextureObject_t CSTable)
{
    printTex<<<1,16>>>(CSTable);
	CUDA_CALL(cudaDeviceSynchronize());
}

__device__ void getDAcs(gFloat e, gFloat *cs, gFloat *valid, cudaTextureObject_t DACSTable)
{
	if (e < 4.5 || e > 12.7)
	{
		*cs = 0.0;
		*valid = 0.0;
	}
	else
	{
		gFloat shift = 10 * e - 43; //DACS_OFFSET minimum energy for DACS
		gFloat entry = (shift / 2) + 0.5;
		*cs = tex2D<gFloat>(DACSTable, entry, 1.5);
		*valid = 1.0;
	}
}

__device__ void getElastCS(gFloat elog, gFloat *cs, gFloat *valid, cudaTextureObject_t ieeCSTable)
{
	gFloat entry = elog * 10 + 0.5;
	*cs = powf(10, tex2D<gFloat>(ieeCSTable, entry, 10.5));
	*valid = 1.0;
}

__device__ void getExcitCS(gFloat elog, gFloat *cs, gFloat *valid, cudaTextureObject_t ieeCSTable)
{
	int i;
	gFloat entry = elog * 10 + 0.5;
	for(i=0; i<5; i++)
	{
		gFloat intp = tex2D<gFloat>(ieeCSTable, entry, (gFloat) i + 5.0 + 0.5); //ieeCSTable[5]
		if (intp <= 0.0) 
		{
			valid[i] = 0.0;
		    cs[i] = 0.0;
		}
		else 
		{
			cs[i] = powf(10, intp); 
			valid[i] = 1.0;
		}
//	    printf("valid = %f, elog = %f, entry = %f, cs = %f\n", valid[i], elog, entry, cs[i]);
	}
}

__device__ void getIonCS(gFloat elog, gFloat *cs, gFloat *valid, cudaTextureObject_t ieeCSTable)
{
	int i;
	gFloat entry = elog * 10 + 0.5;
	for(i=0; i<5; i++)
	{
		gFloat intp = tex2D<gFloat>(ieeCSTable, entry, (gFloat) i + 0 + 0.5); //ieeCSTable[0]
		if (intp <= 0.0)
		{
		    cs[i] = 0.0;
			valid[i] = 0.0;
		}
		else 
		{
	        cs[i] = powf(10, intp); 
			valid[i] = 1.0;
		}
//	    printf("valid = %f, elog = %f, entry = %f, cs = %f\n", valid[i], elog, entry, cs[i]);
	}
}

__device__ void tableIntp(gFloat elog, Particle *electron, gFloat *reacCS, gFloat *reacValid, cudaTextureObject_t ieeCSTable, cudaTextureObject_t DACSTable)
{
	getDAcs(electron->e, &reacCS[11], &reacValid[11], DACSTable);
    getElastCS(elog, &reacCS[10], &reacValid[10], ieeCSTable);
    getExcitCS(elog, &reacCS[5], &reacValid[5], ieeCSTable);
    getIonCS(elog, &reacCS[0], &reacValid[0], ieeCSTable);
	    
}
__device__ void actChoice(curandState *seed, gFloat *reacValid, gFloat *reacCS, Particle *electron_ptr)
/*******************************************************************
c*   Sum up all possible cross sections then random choose one     *
c*   reaction                                                      *
c*                                                                 *
c*    Input:                                                       *
c*      reacCS -> cross section from LUT                           *                                                     
c*      reacValid -> valid bit of cross section value              *
c*    Output:                                                      *
c*      electron_ptr->h2oState -> 0 ~ 4   ionizaton                *
c*                                5 ~ 9   excitation               *
c*                                 10     elastic                  *
c*                                 11     dissociative             *
c******************************************************************/
{
	gFloat csSum = 0.0;
    gFloat ratioCSAdd = 0.0;
    
    // Sum up all possible cross sections
    for(int i=0; i<12; i++)
    {
    	csSum += reacValid[i] * reacCS[i];
//		printf("reacValid[%d]=%f reacCS[%d]=%f\n", i, reacValid[i], i, reacCS[i]);
    }
    
    // Random choose one reaction
    gFloat reacRand = curand_uniform(seed);	
//	printf(" \nreacRand=%f\n", reacRand);
    for(int i=0; i<12; i++)
    {
    	ratioCSAdd += reacValid[i] * reacCS[i] / csSum;
    	if (reacRand < ratioCSAdd)
    	{
    		electron_ptr->h2oState = i;
   		    break;
    	}
    }
	electron_ptr->path = eDistance(seed, csSum);
	//printf(" csSum=%f, path=%e\n", csSum, electron_ptr->path);
}

__device__ gFloat eDistance(curandState *seed, gFloat csSum)
{ //sample according to exp(-s/lamda)

    gFloat lamda = 1 / (1.0*csSum);// free mean length in cm.
    gFloat U = 0.0;
    do
    {
    	U = curand_uniform(seed);
    } while (U == 0.0);   
    return (-1 * lamda * log(U));
}

/*defining two functions */
__device__ gFloat wx(gFloat r, gFloat t, uint c){
    switch (c){
    case 0: return r*(t - 1) / 2;
    case 1: return 1 / (1 - r*(t - 1) / (t + 1)) - 1;
    case 2: return r*t*(t - 1) / ((t + 1) + r*(t - 1));
    case 3: return sqrt(1 / (1 - r*((t + 1)*(t + 1) - 4) / (t + 1) / (t + 1))) - 1;
	default : return r*(t - 1) / 2;
    }
}

__device__ gFloat hx(gFloat t, gFloat w, gFloat tp, uint c){
    switch (c){
    case 0: return 1;
    case 1: return (t + 1) / t*(1 - (w + 1) / (t + 1)*(1 + 2 * tp) / (1 + tp / 2) / (1 + tp / 2));
    case 2: return 2 * (1 - (t - w) / (t + 1));
    case 3: return 0.5*(1 + pow((w + 1) / (t - w), 3));
	default : return 1;
    }
}
/*defining two functions */

__device__ void ionE2nd(gFloat * e2nd, Particle * thisOne, uint channel, curandState *seed)
{
    gFloat B[5] = { 10.79, 13.39, 16.05, 32.30, 539.00 };
    gFloat Bp[5] = { 2.11154598825832e-05, 2.62035225048924e-05, 3.14090019569472e-05, 6.32093933463797e-05, 0.00105479452054795 };

    gFloat bp = Bp[channel];
    gFloat tp = thisOne->e / MCC;

    gFloat betaT2 = 1 - 1 / (1 + tp) / (1 + tp);

    gFloat t = thisOne->e / B[channel];


    gFloat A1 = t*(t - 1) / (t + 1) / (t + 1);
    gFloat A2 = (t - 1) / t / (t + 1) / 2;
    gFloat A3 = (log(betaT2 / (1 - betaT2)) - betaT2 - log(2 * bp))*((t + 1)*(t + 1) - 4) / (t + 1) / (t + 1);
    gFloat A0 = bp*bp*(t - 1) / (1 + tp / 2) / (1 + tp / 2) / 2;
    gFloat sumA = A1 + A2 + A3 + A0;

    gFloat r1, r2, r3, w;
    while (1){
        r1 = curand_uniform(seed) * sumA;
        r2 = curand_uniform(seed);
        r3 = curand_uniform(seed);
        if (r1 < A0){
            w = wx(r2, t, 0);
            if (r3 <= hx(t, w, tp, 0))
                break;
        }
        else if (r1 < A0 + A1){
            w = wx(r2, t, 1);
            if (r3 <= hx(t, w, tp, 1))
                break;
        }
        else if (r1 < A0 + A1 + A2){
            w = wx(r2, t, 2);
            if (r3 <= hx(t, w, tp, 2))
                break;
        }
        else{
            w = wx(r2, t, 3);
            if (r3 <= hx(t, w, tp, 3))
                break;
        }
    }

    thisOne->e -= B[channel] + w*B[channel];
    *e2nd = w*B[channel];
}

__device__ void eDrop(curandState *seed, Particle *electron_ptr, gFloat elog, gFloat *ei, gFloat *e2nd, gFloat *eRatio, cudaTextureObject_t BindE_array, float *edrop)
{
    //int id = blockDim.x*blockIdx.x+threadIdx.x;
	*ei = electron_ptr->e;
    *edrop = 0;
    *e2nd = 0;
	gFloat dE = 0;
  	if (electron_ptr->h2oState < 5)  //ionizaton
    {
    	ionE2nd(e2nd, electron_ptr, electron_ptr->h2oState, seed);			
		*eRatio = *e2nd / *ei;
        *edrop = *ei - electron_ptr->e -*e2nd;// total energy is divided into three parts: remaining, new born, deposited
    }
    else if (electron_ptr->h2oState < 10) //excitation
    {
    	dE = tex1D<gFloat>(BindE_array, (float)electron_ptr->h2oState+0.5);
		*eRatio = dE / *ei;
    	electron_ptr->e -= dE;
        *edrop = dE;
    }
    else if (electron_ptr->h2oState == 11) //dissociative
    {
        *edrop = electron_ptr -> e; // deposits all energy
        electron_ptr->dead = 1;
    }
	if (electron_ptr->e <= eEcutoff ) electron_ptr->dead = 1;  
}

__device__ gFloat elastDCS(curandState *seed, gFloat elog, cudaTextureObject_t elastDCSTable)
{
	gFloat entry = elog * 10 + 0.5;
	gFloat sel = 100.0 * curand_uniform(seed) + 0.5;
	return acos(tex2D<gFloat>(elastDCSTable, entry, sel));
}

__device__ void eAngles(curandState *seed, Particle *electron_ptr, gFloat elog, gFloat ei, gFloat e2nd, gFloat eRatio, gFloat *polar, gFloat *azi, gFloat *polar_e2nd, gFloat *azi_e2nd, cudaTextureObject_t elastDCSTable)
{
    // angles
    if (electron_ptr->h2oState < 5)  //ionizaton
    {
    	*polar = asin(sqrt(eRatio/((1-eRatio) * ei/TWOMCC + 1)));
    	
    	if (e2nd > 200.0)
    	{
    	    *polar_e2nd = asin(sqrt((1-eRatio)/(1+e2nd/TWOMCC)));	    	
    	}
    	else if (e2nd >= 50.0)
    	{
    		if (curand_uniform(seed) > 0.1)
    			*polar_e2nd = PI * (curand_uniform(seed) / 4) + (PI/4);
    		else
    			*polar_e2nd = PI * (curand_uniform(seed));
    	}
    	else if (e2nd > 0)
    	{
    		*polar_e2nd = PI * (curand_uniform(seed));
    	}
    	
    	*azi = 2 * PI * (curand_uniform(seed));
    	*azi_e2nd = *azi - PI;
    }
    else if (electron_ptr->h2oState < 10) //excitation
    {
    	*polar = asin(sqrt(eRatio/((1-eRatio) * ei/TWOMCC + 1)));
    	*azi = 2 * PI * (curand_uniform(seed));
    }
    else if (electron_ptr->h2oState < 11) //elastic
	{
		*polar = elastDCS(seed, elog, elastDCSTable);
		*azi = 2 * PI * (curand_uniform(seed));
	}
}

__device__ void eHop(curandState *seed, Particle *electron_ptr, gFloat polar, gFloat azi)
{// move the electron and change its velocity direction    
    electron_ptr->x += electron_ptr->path*electron_ptr->ux;
    electron_ptr->y += electron_ptr->path*electron_ptr->uy;
    electron_ptr->z += electron_ptr->path*electron_ptr->uz;		
	
	gFloat costh = cos(polar);	
	rotate(&electron_ptr->ux, &electron_ptr->uy, &electron_ptr->uz, costh, azi);  
}

__device__ void eTime(Particle *electron_ptr, gFloat ei)
{
    gFloat v = C * sqrt(1 - M2C4/((ei+MCC)*(ei+MCC))); // m/s
    electron_ptr->elape_time += electron_ptr->path / (100.0f * v); // path(cm) to m, time unit in second
}

__device__ void e2ndQ(Particle *electron_ptr, float edrop, Data *container, Particle *e2Queue, gFloat e2nd, int *e2nd_num, gFloat polar_e2nd, gFloat azi_e2nd, int *where, int *gEid)
{
    //int id = blockDim.x*blockIdx.x+threadIdx.x;
	if (electron_ptr->h2oState < 5)
	{
		int i = atomicAdd(e2nd_num, 1);

        e2Queue[i].ux = sin(polar_e2nd)*cos(azi_e2nd);// 2nd e- born
        e2Queue[i].uy = sin(polar_e2nd)*sin(azi_e2nd);
        e2Queue[i].uz = cos(polar_e2nd);               
        e2Queue[i].x = electron_ptr->x;
        e2Queue[i].y = electron_ptr->y;
        e2Queue[i].z = electron_ptr->z;
	    e2Queue[i].dead = 0;
	    e2Queue[i].e = e2nd;
	    e2Queue[i].h2oState = 99;//electron_ptr->h2oState;
        e2Queue[i].elape_time = electron_ptr->elape_time;
		int j = atomicAdd(gEid, 1);
        e2Queue[i].id = j;
	    //e2Queue[i].parentId = electron_ptr->parentId;			
        e2Queue[i].parentId = electron_ptr->id;
        e2Queue[i].ptype = -1;
        e2Queue[i].A = 1/1897.0;
	
	// Save parent ionization state	
		int k = atomicAdd(where, 1);    
        container[k].x = electron_ptr->x;
        container[k].y = electron_ptr->y;
        container[k].z = electron_ptr->z;
        container[k].e = edrop;
        container[k].time = electron_ptr->elape_time;
        container[k].h2oState = electron_ptr->h2oState;
        container[k].id = electron_ptr->id;			
        container[k].parentId = electron_ptr->parentId;			
	}	
}
 
__device__ void eRecord(Particle *electron_ptr, Data *container, float edrop, int *where, int ContainerSize)
{
    if(*where > 0.99*ContainerSize) printf("container may overflow... %d\n", *where);
    //int id = blockDim.x*blockIdx.x+threadIdx.x;
    if (electron_ptr->h2oState < 5 || electron_ptr->h2oState == 99)
    {
		if (electron_ptr->e <= eEcutoff)      // 1. Generates e2nd after ionization reaction then dead (< cut-off) in the same step
		{                                        // 2. New born electron 
		    
            int k = atomicAdd(where, 1);
            electron_ptr->h2oState = -1;
	        container[k].x = electron_ptr->x;
	        container[k].y = electron_ptr->y;
	        container[k].z = electron_ptr->z;
	        container[k].e = electron_ptr->e;  
                      
	        container[k].time = electron_ptr->elape_time;
		    container[k].h2oState = electron_ptr->h2oState;
            //if(id==0) printf("cutoff record energy %f h2oState %d\n", container[k].e, container[k].h2oState);
	        container[k].id = electron_ptr->id;
	        container[k].parentId = electron_ptr->parentId;	

		    electron_ptr->e = 0;
			electron_ptr->dead = 1;
			//if (electron_ptr->h2oState == -1 || electron_ptr->h2oState == 11) printf("Eid1 = %5d, state = %d, e = %e, k = %d\n", electron_ptr->id, electron_ptr->h2oState, electron_ptr->e, k);
		}
	}			
	else if (electron_ptr->h2oState != 10) // other events happened
    {
        int k = atomicAdd(where, 1);
        container[k].x = electron_ptr->x;
        container[k].y = electron_ptr->y;
        container[k].z = electron_ptr->z;
        container[k].e = edrop;
        container[k].time = electron_ptr->elape_time;
    	container[k].h2oState = electron_ptr->h2oState;
        //if(id==0) printf("excitation record energy %f h2oState %d\n", container[k].e, container[k].h2oState);
        container[k].id = electron_ptr->id;
        container[k].parentId = electron_ptr->parentId;			
    	//if (electron_ptr->h2oState == -1 || electron_ptr->h2oState == 11) printf("Eid2 = %5d, state = %d, e = %e, k = %d\n", electron_ptr->id, electron_ptr->h2oState, electron_ptr->e, k);
    }

	// excitation dead : record when both excitation state & < cut-off in the same step
	if (electron_ptr->h2oState > 4 && electron_ptr->h2oState < 10 && electron_ptr->dead == 1)
    {
        int k = atomicAdd(where, 1);
    	electron_ptr->h2oState = -1;
        container[k].x = electron_ptr->x;
        container[k].y = electron_ptr->y;
        container[k].z = electron_ptr->z;
        container[k].e = electron_ptr->e;        
        container[k].time = electron_ptr->elape_time;
    	container[k].h2oState = electron_ptr->h2oState;
        //if(id==0) printf("excitation dead record energy %f h2oState %d\n", container[k].e, container[k].h2oState);
        container[k].id = electron_ptr->id;
        container[k].parentId = electron_ptr->parentId;			
    	//if (electron_ptr->h2oState == -1 || electron_ptr->h2oState == 11) printf("Eid3 = %5d, state = %d, e = %e, k = %d\n", electron_ptr->id, electron_ptr->h2oState, electron_ptr->e, k);
    }
}

__global__  void eTransport(int N, int ContainerSize, Particle *eQueue, Particle *e2Queue, Data *container,
          int *where, int *second_num, int *gEid,
		  cudaTextureObject_t DACSTable, cudaTextureObject_t BindE_array, 
		  cudaTextureObject_t ieeCSTable, cudaTextureObject_t elastDCSTable, int MaxN)
{// kernel for electron transport
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int iniid = id;
    curandState seed = cuseed[iniid];
    
	while (id < N)
	{
	    gFloat ei = 0.0;
	    gFloat e2nd = 0.0;
	    gFloat eRatio = 0.0;
	    gFloat polar = 0.0;
	    gFloat polar_e2nd = 0.0;
	    gFloat azi = 0.0;
	    gFloat azi_e2nd = 0.0;
        gFloat edrop;	    
	    gFloat elog;
	    gFloat reacCS[12];
	    gFloat reacValid[12];
        Particle electron= eQueue[id]; // copy to local memory for faster process
		
        int localsecCal = 0;
        while (electron.dead == 0)
        {
            e2nd = 0;
            edrop = 0;
            if(boundaryShape>=0)
            {
                //if(id<5) printf("Boundary check!\n");
                if(applyBoundary(boundaryShape,boundaryCenter, boundarySize, electron.x, electron.y, electron.z)) break;
            }

			if(electron.e < 0)
				break;

			if (electron.e > eEcutoff)
			{
				gFloat ein = electron.e;               
                do
		        {
                    electron.e = ein;
                    electron.dead = 0;
                    elog = log10(ein);
                    tableIntp(elog, &electron, reacCS, reacValid, ieeCSTable, DACSTable);
                    actChoice(&seed, reacValid, reacCS, &electron);	
                    if(electron.h2oState !=10) eDrop(&seed, &electron, elog, &ei, &e2nd, &eRatio, BindE_array, &edrop);						
		        } while (electron.e < 0);
                eAngles(&seed, &electron, elog, ei, e2nd, eRatio, &polar, &azi, &polar_e2nd, &azi_e2nd, elastDCSTable);
                eHop(&seed, &electron, polar, azi);
                eTime(&electron, ei);
                e2ndQ(&electron, edrop, container, e2Queue, e2nd, second_num, polar_e2nd, azi_e2nd, where, gEid);
			}
            if(electron.h2oState<5) localsecCal += 1;
            eRecord(&electron, container, edrop, where, ContainerSize);

           // printf("electron %d drops %f energy produce secondary elec %f\n", electron.id, edrop, e2nd);
		}
        
        //printf("electron %d produce %d secondaries\n", electron.id, localsecCal);
        id += blockDim.x*gridDim.x;	
	}
    cuseed[iniid] = seed;
    
}


__device__ void actChoice_proton(curandState *seed, cudaTextureObject_t protonCSTable, Particle *particle_ptr)
{
	gFloat csSum = 0.0;
    gFloat ratioCSAdd = 0.0;
    float elog = log10(particle_ptr->e/particle_ptr->A);
    float entry = (elog-1.0)/0.05+0.5; //hard coding, related to smallest proton energy 10^1 eV, interval 10^0.05 in table
    float csArray[10]; //only ionization and excitation
    // Sum up all possible cross sections
    for(int i=0; i<10; i++)
    {
    	csArray[i] = tex2D<gFloat>(protonCSTable, entry, (gFloat) i + 0.5);
        csSum += csArray[i];
    }
    
    // Random choose one reaction
    gFloat reacRand = curand_uniform(seed);	
    for(int i=0; i<10; i++)
    {
    	ratioCSAdd += csArray[i] / csSum;
    	if (reacRand < ratioCSAdd)
    	{
    		particle_ptr->h2oState = i;
   		    break;
    	}
    }
    float beta = sqrt(1-1/pow(1+particle_ptr->e/particle_ptr->A/Mpcc,2)); //relativistic beta
    float x = 100*beta*pow(particle_ptr->ptype/1.0,-2.0/3.0);
    float correctZ = 1- exp(-1.316*x+0.112*x*x-0.065*x*x*x);    
    correctZ = particle_ptr->ptype*correctZ;//zeff

    float lamda = 1/csSum/correctZ/correctZ; // related to eDistance function
    float rando;
    do
    {
        rando = curand_uniform(seed);
    } while (rando == 0.0);
    
    particle_ptr->path = (-1 * lamda * log(rando)); //step-length 
}

__device__ float sampleWIon(curandState *seed, Particle *particle_ptr)
{
    // sampling energy of secondary electron for ionization channnel for equavalent proton energy > 500 keV

    if(particle_ptr->e/particle_ptr->A < I[particle_ptr->h2oState])
        return 0;
    float T = mecc*particle_ptr->e/particle_ptr->A/Mpcc;
    float v = sqrt(mecc*0.5/I[particle_ptr->h2oState]*(1-1/pow(1+T/mecc,2)));
    //if(id ==0 ) printf("required parameters %f %f\n", T, v);
    int index = particle_ptr->h2oState<1?0:1;
    float *para = &parameters[index*9];
    float f1 = para[2]*pow(v,para[3])/(1+para[4]*pow(v,(para[3]+4)))+para[0]*log(1+pow(v,2))/(pow(v,2)+para[1]/pow(v,2));
    float f2 = para[7]*pow(v,para[8])*(para[5]/pow(v,2)+para[6]/pow(v,4))/(para[7]*pow(v,para[8])+para[5]/pow(v,2)+para[6]/pow(v,4));
    //if(id ==0 ) printf("required parameters %f %f\n", f1, f2);
    float wi = 4*pow(v,2)-2*v-Ryd*0.25/I[particle_ptr->h2oState];
    float wmax = particle_ptr->e/particle_ptr->A/I[particle_ptr->h2oState]-1;
    //if(id ==0 ) printf("required parameters %f %f\n", wi, wmax);
    float c = wmax*(f2*wmax+f1*(2+wmax))*0.5/pow((1+wmax),2);
    //if(id ==0 ) printf("required parameters %f\n", c);
    float rando, ws, real;
    
    do
    {
        rando = curand_uniform(seed);
        ws = (-f1+2*c*rando+sqrt(pow(f1,2)+2*f2*rando*c-2*f1*rando*c))/(f1+f2-2*c*rando);
        real = 1/(1+exp(alpha[index]*(ws-wi)/v));
        //if(id ==0 ) printf("required parameters %f %f\n", ws, real);
        rando = curand_uniform(seed);
    }while(rando > real);

    return ws*I[particle_ptr->h2oState];    
}

__device__ float sampleWExc(curandState *seed, Particle *particle_ptr)
{
    // sample energy loss for excitation for equavalent proton energy > 500 keV
    int channel = particle_ptr->h2oState-5; // change offset
    float T = mecc*particle_ptr->e/particle_ptr->A/Mpcc;
    //float T = mecc*(1-1/pow((1+particle_ptr->e/particle_ptr->A/Mpcc),2))/2;
    float Wmin = 2;
    float Wmax = 50;
    float u1 = 1+exp(alphai[channel]*(Wmin-Woi[channel]));
    float u2 = 1+exp(alphai[channel]*(Wmax-Woi[channel]));
    float c = (1/u1-1/u2)/alphai[channel];
    //if(id ==0 ) printf("required parameters %f %f %f\n", u1,u2,c);
    float Ws, rando, real;
    if(channel == 2)
    {
        do
        {
            rando = curand_uniform(seed);
            Ws = Woi[channel] + log(u1/(1-alphai[channel]*u1*rando*c)-1)/alphai[channel]; 
            real = log(4*T/Ws)/Ws;
            rando = curand_uniform(seed)*log(4*T/Wmin)/Wmin;
        }while(rando > real);
    }
    else
    {
        do
        {
            Ws = Woi[channel]+curand_normal(seed)/sqrt(2*alphai[channel]); 
            real = log(4*T/Ws)/Ws;
            rando = curand_uniform(seed)*log(4*T/Wmin)/Wmin;
        }while(rando > real);
    }
    return Ws;
}

__device__ void eDrop_proton(curandState *seed, Particle *particle_ptr, float *e2nd, float *edrop)
{
	gFloat dE = 0;
    float B[5] = {8.17, 10.13, 11.31, 12.91, 14.5};
    //int id = blockDim.x*blockIdx.x+threadIdx.x;
  	if (particle_ptr->h2oState < 5)  //ionizaton
    {
        *e2nd = sampleWIon(seed, particle_ptr);
        dE = (*e2nd) + I[particle_ptr->h2oState];
        *edrop = I[particle_ptr->h2oState];
    }
    else if (particle_ptr->h2oState < 8 && particle_ptr->e/particle_ptr->A>5e5f) //excitation
    {
    	dE = sampleWExc(seed, particle_ptr);
        *edrop = dE;  	
    }
    else
    {
        dE = B[particle_ptr->h2oState-5];
        *edrop = dE;
    }
	particle_ptr->e -= dE;
 	if (particle_ptr->e/particle_ptr->A <= pECutoff) particle_ptr->dead = 1;
}


__device__ void eAngles_proton(curandState *seed, Particle *particle_ptr, gFloat ei, gFloat e2nd, gFloat *polar, gFloat *azi, gFloat *polar_e2nd, gFloat *azi_e2nd)
{
    // angles
    if (particle_ptr->h2oState < 5)  //ionizaton
    {    	
    	if (e2nd > I[particle_ptr->h2oState])
    	{
    	    *polar_e2nd = acos(sqrt(e2nd*0.25/ei));	    	
    	}
    	else 
    	{   	
    		*polar_e2nd = PI * (curand_uniform(seed));
    	}
    	
    	*azi_e2nd = 2 * PI * (curand_uniform(seed));
    }
    if(particle_ptr->h2oState == 13)
    {
        *polar_e2nd = PI * (curand_uniform(seed));
        *azi_e2nd = 2 * PI * (curand_uniform(seed));
    }
    *polar = 0; // no deviation for proton and heavy ions
    *azi = 0;
}


__device__ void eHop_proton(curandState *seed, Particle *particle_ptr, gFloat polar, gFloat azi, float ei)
{  
    
    particle_ptr->x += particle_ptr->path*particle_ptr->ux;
    particle_ptr->y += particle_ptr->path*particle_ptr->uy;
    particle_ptr->z += particle_ptr->path*particle_ptr->uz;	

    float gamma = ei/particle_ptr->A/Mpcc+1;
	float v = C * sqrt(1 - 1/pow(gamma,2));
	particle_ptr->elape_time += particle_ptr->path / (100.0 * v); // adding time

	gFloat costh = cos(polar);	
	rotate(&particle_ptr->ux, &particle_ptr->uy, &particle_ptr->uz, costh, azi);  
}

__global__  void pTransport(int N, int ContainerSize, Particle *pQueue, Particle *e2Queue, Data *container, int *where, int *second_num, int *gEid,
    cudaTextureObject_t protonCSTable, int MaxN)
{
    // kernel for transport of protons and heavy ions
    
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int iniid = id;
    curandState seed = cuseed[iniid];
    while (id < N)
    {
        gFloat ein = 0.0;
        gFloat e2nd = 0.0;

        gFloat polar = 0.0;
        gFloat polar_e2nd = 0.0;
        gFloat azi = 0.0;
        gFloat azi_e2nd = 0.0;
        float edrop;
        Particle particle = pQueue[id];
        if(id<10) printf("particle.parentId = %d particle.e is %f\n", particle.parentId, particle.e);
        while (particle.dead == 0)
        {
            e2nd = 0;
            edrop =0;
            if(boundaryShape>=0)
            {
                if(applyBoundary(boundaryShape,boundaryCenter, boundarySize, particle.x, particle.y, particle.z)) break;
            }
            if(particle.e <= 0)
                break;

            if (particle.e > pECutoff*particle.A)
            {
                ein = particle.e;
                do
                {
                    particle.e = ein;
                    particle.dead = 0;
                    actChoice_proton(&seed, protonCSTable, &particle);	
                    eDrop_proton(&seed, &particle, &e2nd, &edrop);						
                    // if(id == 0) printf("ein %f, h2oState = %d, edrop is %f, eout %f\n", ein, particle.h2oState,ein-particle.e, particle.e);
                } while (particle.e < 0);
                eAngles_proton(&seed, &particle, ein, e2nd,  &polar, &azi, &polar_e2nd, &azi_e2nd);
                eHop_proton(&seed,  &particle, polar, azi,ein);
                e2ndQ(&particle, edrop, container, e2Queue, e2nd, second_num, polar_e2nd, azi_e2nd, where, gEid);
            }
            eRecord(&particle, container,  edrop, where, ContainerSize);
        }
        id += blockDim.x*gridDim.x;
    }
    cuseed[iniid] = seed;
}