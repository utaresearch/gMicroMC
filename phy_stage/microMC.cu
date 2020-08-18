#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "microMC.h"
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#include <thrust/device_vector.h> 
#include <thrust/sort.h>
#include <thrust/device_ptr.h>


void rd_dacs(FILE *, REAL **);
void rd_ioncs(FILE *, REAL *, REAL **);
void rd_iondcs(FILE *, FILE *, FILE *, FILE *, FILE *, 
	           REAL **, REAL **, REAL **, REAL **, REAL **);
void rd_elast(FILE *, FILE *, REAL **, REAL **);
void rd_excit(FILE *, REAL *, REAL **);
void getElastCS(REAL, REAL *, REAL *, REAL **);
void getExcitCS(REAL, REAL *, REAL *, REAL **);
void getIonCS(REAL, REAL *, REAL *, REAL **);

__constant__ REAL negcub_len_dev;
__constant__ REAL cub_len_dev;
__constant__ REAL Ecutoff_dev;

struct compare_eStruct{
	__host__ __device__ bool operator()(eStruct a,eStruct b){return (a.e > b.e) ? false : true;}
};

void cudaCHECK()
{
    cudaError_t errSync  = cudaGetLastError();
    cudaError_t errAsync = cudaDeviceSynchronize();
    if (errSync != cudaSuccess) 
      printf("\n Sync kernel error: %s\n", cudaGetErrorString(errSync));
    if (errAsync != cudaSuccess)
      printf(" Async kernel error: %s\n", cudaGetErrorString(errAsync));
}

__device__
void getDAcs(REAL e, REAL *cs, REAL *valid, cudaTextureObject_t DACSTable)
{
	if (e < 4.5 || e > 12.7)
	{
		*cs = 0.0;
		*valid = 0.0;
	}
	else
	{
		REAL shift = 10 * e - DACS_OFFSET;
		REAL entry = (shift / 2) + 0.5;
		*cs = tex2D<REAL>(DACSTable, entry, 1.5);
		*valid = 1.0;
	}
}

__device__
void getElastCS(REAL elog, REAL *cs, REAL *valid, cudaTextureObject_t ieeCSTable)
{
	REAL entry = elog * 10 + 0.5;
	*cs = powf(10, tex2D<REAL>(ieeCSTable, entry, 10.5));
	*valid = 1.0;
}

__device__
void getExcitCS(REAL elog, REAL *cs, REAL *valid, cudaTextureObject_t ieeCSTable)
{
	int i;
	REAL entry = elog * 10 + 0.5;
	for(i=0; i<BINDING_ITEMS; i++)
	{
		REAL intp = tex2D<REAL>(ieeCSTable, entry, (REAL) i + 5.0 + 0.5); //ieeCSTable[5]
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

__device__
void getIonCS(REAL elog, REAL *cs, REAL *valid, cudaTextureObject_t ieeCSTable)
{
	int i;
	REAL entry = elog * 10 + 0.5;
	for(i=0; i<BINDING_ITEMS; i++)
	{
		REAL intp = tex2D<REAL>(ieeCSTable, entry, (REAL) i + 0 + 0.5); //ieeCSTable[0]
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

__device__
void tableIntp(REAL elog, eStruct *electron, REAL *reacCS, REAL *reacValid, cudaTextureObject_t ieeCSTable, cudaTextureObject_t DACSTable)
{
	getDAcs(electron->e, &reacCS[11], &reacValid[11], DACSTable);
    getElastCS(elog, &reacCS[10], &reacValid[10], ieeCSTable);
    getExcitCS(elog, &reacCS[5], &reacValid[5], ieeCSTable);
    getIonCS(elog, &reacCS[0], &reacValid[0], ieeCSTable);
	    
}

__global__ void rand_init(unsigned int seed, CURANDSTATE* states)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, /* the seed can be the same for each core, here we pass the time in from the CPU */
                i, /* the sequence number should be different for each core (unless you want all
                             cores to get the same sequence of numbers for some reason - use thread id! */
                0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
                &states[i]);
}


__device__
void launchE(eStruct eQueue, eStruct *electron)
{
    electron->x          = eQueue.x          ;
    electron->y          = eQueue.y          ;
    electron->z          = eQueue.z          ;      
    electron->ux         = eQueue.ux         ;
    electron->uy         = eQueue.uy         ;
    electron->uz         = eQueue.uz         ;      
    electron->e          = eQueue.e          ;
    electron->h2oState   = eQueue.h2oState   ;
    electron->dead       = eQueue.dead       ;
    electron->path       = eQueue.path       ;	
    electron->id         = eQueue.id         ;	
    electron->parentId   = eQueue.parentId   ;	
    electron->elape_time = eQueue.elape_time ;	
}

__device__
REAL eDistance(CURANDSTATE *seed, REAL csSum)
{
    // distance s
    REAL lamda = 1 / (1.0*csSum);//Revised at May 20th, change the cross section so that the free mean length changes.
    REAL U = 0.0;
    do
    {
    	U = curand_uniform(seed);
    } while (U == 0.0);
    
//	printf(" lamda=%e, U=%e, g_seed=%u\n", lamda, U, g_seed);
    return (-1 * lamda * log(U));
}


__device__
void actChoice(CURANDSTATE *seed, REAL *reacValid, REAL *reacCS, eStruct *electron_ptr)
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
	REAL csSum = 0.0;
    REAL ratioCSAdd = 0.0;

//     printf("== rand =%f\n", curand_uniform(seed));
    // Sum up all possible cross sections
    for(int i=0; i<INTERACTION_TYPES; i++)
    {
    	csSum += reacValid[i] * reacCS[i];
//		printf("reacValid[%d]=%f reacCS[%d]=%f\n", i, reacValid[i], i, reacCS[i]);
    }
    
    // Random choose one reaction
    REAL reacRand = curand_uniform(seed);	
//	printf(" \nreacRand=%f\n", reacRand);
    for(int i=0; i<INTERACTION_TYPES; i++)
    {
		//REAL cs = reacCS[i];
		
		//if (i == 5) cs = 0.1*cs;
		
    	ratioCSAdd += reacValid[i] * reacCS[i] / csSum;
    	if (reacRand < ratioCSAdd)
    	{
    		electron_ptr->h2oState = i;
//if (electron_ptr->h2oState!=10){
//  printf("h2oState = %d\n", electron_ptr->h2oState);
//  printf("reacCS[%d]=%f reacValid=%f\n", i, reacCS[i], reacValid[i]);
//}
   		    break;
    	}
    }
	electron_ptr->path = eDistance(seed, csSum);
	//printf(" csSum=%f, path=%e\n", csSum, electron_ptr->path);
}

__device__
REAL wx(REAL r, REAL t, uint c){
    switch (c){
    case 0: return r*(t - 1) / 2;
    case 1: return 1 / (1 - r*(t - 1) / (t + 1)) - 1;
    case 2: return r*t*(t - 1) / ((t + 1) + r*(t - 1));
    case 3: return sqrt(1 / (1 - r*((t + 1)*(t + 1) - 4) / (t + 1) / (t + 1))) - 1;
	default : return r*(t - 1) / 2;
    }
}

__device__
REAL hx(REAL t, REAL w, REAL tp, uint c){
    switch (c){
    case 0: return 1;
    case 1: return (t + 1) / t*(1 - (w + 1) / (t + 1)*(1 + 2 * tp) / (1 + tp / 2) / (1 + tp / 2));
    case 2: return 2 * (1 - (t - w) / (t + 1));
    case 3: return 0.5*(1 + pow((w + 1) / (t - w), 3));
	default : return 1;
    }
}

__device__
void ionE2nd(REAL * e2nd, eStruct * thisOne, uint channel, CURANDSTATE *seed){
    REAL B[5] = { 10.79, 13.39, 16.05, 32.30, 539.00 };
    REAL Bp[5] = { 2.11154598825832e-05, 2.62035225048924e-05, 3.14090019569472e-05, 6.32093933463797e-05, 0.00105479452054795 };

    REAL bp = Bp[channel];
    REAL tp = thisOne->e / MCC;

    REAL betaT2 = 1 - 1 / (1 + tp) / (1 + tp);

    REAL t = thisOne->e / B[channel];


    REAL A1 = t*(t - 1) / (t + 1) / (t + 1);
    REAL A2 = (t - 1) / t / (t + 1) / 2;
    REAL A3 = (log(betaT2 / (1 - betaT2)) - betaT2 - log(2 * bp))*((t + 1)*(t + 1) - 4) / (t + 1) / (t + 1);
    REAL A0 = bp*bp*(t - 1) / (1 + tp / 2) / (1 + tp / 2) / 2;
    REAL sumA = A1 + A2 + A3 + A0;

    REAL r1, r2, r3, w;
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

__device__
void eDrop(CURANDSTATE *seed, eStruct *electron_ptr, REAL elog, REAL *ei, REAL *e2nd, REAL *eRatio, cudaTextureObject_t BindE_array)
{
	*ei = electron_ptr->e;
	REAL dE = 0;
  	if (electron_ptr->h2oState < 5)  //ionizaton
    {
    	ionE2nd(e2nd, electron_ptr, electron_ptr->h2oState, seed);			
		*eRatio = *e2nd / *ei;
//		printf(" e2nd=%e, eRatio=%e, electron_ptr->e=%e\n", *e2nd, *eRatio, electron_ptr->e);
    }
    else if (electron_ptr->h2oState < 10) //excitation
    {
    	dE=tex1D<REAL>(BindE_array, (float)electron_ptr->h2oState+0.5);
		*eRatio = dE / *ei;
    	electron_ptr->e-=dE;
    }
	
 	if (electron_ptr->e <= Ecutoff_dev || electron_ptr->h2oState == 11) electron_ptr->dead = 1;
}

__device__
REAL elastDCS(CURANDSTATE *seed, REAL elog, cudaTextureObject_t elastDCSTable)
{
	REAL entry = elog * 10 + 0.5;
	REAL sel = 100.0 * curand_uniform(seed) + 0.5;
	return acos(tex2D<REAL>(elastDCSTable, entry, sel));
}

__device__
void eAngles(CURANDSTATE *seed, eStruct *electron_ptr, REAL elog, REAL ei, REAL e2nd, REAL eRatio, REAL *polar, REAL *azi, REAL *polar_e2nd, REAL *azi_e2nd, cudaTextureObject_t elastDCSTable)
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



__device__ void rotate(REAL *u, REAL *v, REAL *w, REAL costh, REAL phi)
/*******************************************************************
c*    Rotates a vector; the rotation is specified by giving        *
c*    the polar and azimuthal angles in the "self-frame", as       *
c*    determined by the vector to be rotated.                      *
c*                                                                 *
c*    Input:                                                       *
c*      (u,v,w) -> input vector (=d) in the lab. frame             *
c*      costh -> cos(theta), angle between d before and after turn *
c*      phi -> azimuthal angle (rad) turned by d in its self-frame *
c*    Output:                                                      *
c*      (u,v,w) -> rotated vector components in the lab. frame     *
c*    Comments:                                                    *
c*      -> (u,v,w) should have norm=1 on input; if not, it is      *
c*         renormalized on output, provided norm>0.                *
c*      -> The algorithm is based on considering the turned vector *
c*         d' expressed in the self-frame S',                      *
c*           d' = (sin(th)cos(ph), sin(th)sin(ph), cos(th))        *
c*         and then apply a change of frame from S' to the lab     *
c*         frame. S' is defined as having its z' axis coincident   *
c*         with d, its y' axis perpendicular to z and z' and its   *
c*         x' axis equal to y'*z'. The matrix of the change is then*
c*                   / uv/rho    -v/rho    u \                     *
c*          S ->lab: | vw/rho     u/rho    v |  , rho=(u^2+v^2)^0.5*
c*                   \ -rho       0        w /                     *
c*      -> When rho=0 (w=1 or -1) z and z' are parallel and the y' *
c*         axis cannot be defined in this way. Instead y' is set to*
c*         y and therefore either x'=x (if w=1) or x'=-x (w=-1)    *
c******************************************************************/
{
    float rho2,sinphi,cosphi,sthrho,urho,vrho,sinth,norm;

    rho2 = (*u)*(*u)+(*v)*(*v);
    norm = rho2 + (*w)*(*w);
//      Check normalization:
    if (fabs(norm-1.0) > SZERO)
    {
//      Renormalize:
        norm = 1.0/__fsqrt_rn(norm);
        *u = (*u)*norm;
        *v = (*v)*norm;
        *w = (*w)*norm;
    }

    sinphi = __sinf(phi);
    cosphi = __cosf(phi);
//      Case z' not= z:

	float temp = costh*costh;
    if (rho2 > ZERO)
    {
        if(temp < 1.0f)
			sthrho = __fsqrt_rn((1.00-temp)/rho2);
		else 
			sthrho = 0.0f;

        urho =  (*u)*sthrho;
        vrho =  (*v)*sthrho;
        *u = (*u)*costh - vrho*sinphi + (*w)*urho*cosphi;
        *v = (*v)*costh + urho*sinphi + (*w)*vrho*cosphi;
        *w = (*w)*costh - rho2*sthrho*cosphi;
    }
    else
//      2 especial cases when z'=z or z'=-z:
    {
		if(temp < 1.0f)			
	        sinth = __fsqrt_rn(1.00-temp);
		else
			sinth = 0.0f;

        *v = sinth*sinphi;
        if (*w > 0.0)
        {
            *u = sinth*cosphi;
            *w = costh;
        }
        else
        {
            *u = -sinth*cosphi;
            *w = -costh;
        }
    }
}

__device__
void eHop(CURANDSTATE *seed, eStruct *electron_ptr, REAL polar, REAL azi)
{  
    
    electron_ptr->x += electron_ptr->path*electron_ptr->ux;
    electron_ptr->y += electron_ptr->path*electron_ptr->uy;
    electron_ptr->z += electron_ptr->path*electron_ptr->uz;	
	
	
	REAL costh = cos(polar);
	
	rotate(&electron_ptr->ux, &electron_ptr->uy, &electron_ptr->uz, costh, azi);

    //electron_ptr->ux = sin(*polar)*cos(*azi);
    //electron_ptr->uy = sin(*polar)*sin(*azi);
    //if (curand_uniform(seed) > 0.2)
    //    electron_ptr->uz = cos(*polar);
    //else
    //    electron_ptr->uz = -cos(*polar);
    
}

__device__
void eTime(eStruct *electron_ptr, REAL ei)
{
    // electron velocity : time
    REAL v = C * sqrt(1 - M2C4/((ei+MCC)*(ei+MCC))); // m/s
    electron_ptr->elape_time += electron_ptr->path / (100 * v); // path(cm) to m	
}

__device__
void e2ndQ(eStruct *electron_ptr, data *container, eStruct *e2Queue, REAL e2nd, int *e2nd_num, REAL polar_e2nd, REAL azi_e2nd, int *where, int *gEid)
{
    	if (electron_ptr->h2oState < 5)
		{
			int i = atomicAdd(e2nd_num, 1);
            e2Queue[i].ux = sin(polar_e2nd)*cos(azi_e2nd);
            e2Queue[i].uy = sin(polar_e2nd)*sin(azi_e2nd);
            e2Queue[i].uz = cos(polar_e2nd);
                    
            e2Queue[i].x = electron_ptr->x;
            e2Queue[i].y = electron_ptr->y;
            e2Queue[i].z = electron_ptr->z;
		    e2Queue[i].dead = 0;
		    e2Queue[i].e = e2nd;
		    e2Queue[i].h2oState = electron_ptr->h2oState;
            e2Queue[i].elape_time = electron_ptr->elape_time;
			int j = atomicAdd(gEid, 1);
            e2Queue[i].id = j;
 	        //e2Queue[i].parentId = electron_ptr->parentId;			
            e2Queue[i].parentId = electron_ptr->id;
		
		// Save parent ionization state	
			int k = atomicAdd(where, 1);    // 2nd e- born
	        container[k].x = electron_ptr->x;
	        container[k].y = electron_ptr->y;
	        container[k].z = electron_ptr->z;
	        container[k].e = electron_ptr->e;
	        container[k].time = electron_ptr->elape_time;
	        container[k].h2oState = electron_ptr->h2oState;
	        container[k].id = electron_ptr->id;			
	        container[k].parentId = electron_ptr->parentId;			
		}
	
}
 
__device__
void eRecord(eStruct *electron_ptr, data *container, int *where, int ContainerSize)
{
    if(*where > 0.99*ContainerSize) printf("container may overflow... %d\n", *where);
    if (electron_ptr->h2oState < 5 || electron_ptr->h2oState == 99)
    {
		if (electron_ptr->e <= Ecutoff_dev)      // 1. Generates e2nd after ionization reaction then dead (< cut-off) in the same step
		{                                        // 2. New born electron 
		    
            int k = atomicAdd(where, 1);
            electron_ptr->h2oState = -1;
	        container[k].x = electron_ptr->x;
	        container[k].y = electron_ptr->y;
	        container[k].z = electron_ptr->z;
	        container[k].e = electron_ptr->e;
	        container[k].time = electron_ptr->elape_time;
		    container[k].h2oState = electron_ptr->h2oState;
	        container[k].id = electron_ptr->id;
	        container[k].parentId = electron_ptr->parentId;			
		    
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
        container[k].e = electron_ptr->e;
        container[k].time = electron_ptr->elape_time;
    	container[k].h2oState = electron_ptr->h2oState;
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
        container[k].id = electron_ptr->id;
        container[k].parentId = electron_ptr->parentId;			
    	//if (electron_ptr->h2oState == -1 || electron_ptr->h2oState == 11) printf("Eid3 = %5d, state = %d, e = %e, k = %d\n", electron_ptr->id, electron_ptr->h2oState, electron_ptr->e, k);
    }
}


__global__ 
void eTransport(int N, int ContainerSize, eStruct *eQueue, eStruct *e2Queue, int *stepQ, int *cntQ, data *container, CURANDSTATE* states,
          int *where, int *second_num, int *gEid,
		  cudaTextureObject_t DACSTable, cudaTextureObject_t BindE_array, 
		  cudaTextureObject_t ieeCSTable, cudaTextureObject_t elastDCSTable, int iRun, int MaxN)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N)
	{
	    REAL ei = 0.0;
	    REAL e2nd = 0.0;
	    REAL eRatio = 0.0;
	    REAL polar = 0.0;
	    REAL polar_e2nd = 0.0;
	    REAL azi = 0.0;
	    REAL azi_e2nd = 0.0;
        /* curand works like rand - except that it takes a state as a parameter */
#if ACTIVE_SEED == 1
		int mx = MAX_ACTIVE_THRD;
		int seat = i % mx;
		CURANDSTATE seed;
        if (MaxN > MAX_ACTIVE_THRD)
		{
	        seed = states[seat];
		}
		else
		{
	        seed = states[i];
		}
#else
	    CURANDSTATE seed = states[i];
#endif
	    eStruct electron;
	    REAL elog;
	    REAL reacCS[INTERACTION_TYPES];
	    REAL reacValid[INTERACTION_TYPES];
		int step = 0;
		int elascnt = 0;
	    launchE(eQueue[i], &electron);
		//printf("electron.parentId = %d\n", electron.parentId);
		//printf("  id = %5d, electron.dead = %d, electron.e = %e\n", electron.id, electron.dead, electron.e);
        while (electron.dead == 0)
        {
			step++;
			if(electron.h2oState == 10) elascnt++;
			if(iRun == 1 && electron.e < 0 )
				break;
			
			if (electron.e > Ecutoff_dev)
			{
				REAL ein = electron.e;
//	printf("ein = %f\n", ein);
                do
		        {
                    electron.e = ein;
                    electron.dead = 0;
                    elog = log10(ein);
//	printf("elog = %f\n", elog);
                    tableIntp(elog, &electron, reacCS, reacValid, ieeCSTable, DACSTable);
                    actChoice(&seed, reacValid, reacCS, &electron);	
                    eDrop(&seed, &electron, elog, &ei, &e2nd, &eRatio, BindE_array);						
					//printf(" h2oState = %d\n", electron.h2oState);
		        } while (electron.e < 0);
//		if (electron.h2oState < 5) printf("  i=%d, e2nd=%e\n", i, e2nd);
//                if (electron.h2oState != 10)
//				{
//					printf(" e = %e\n", electron.e);
//					printf(" h2oState = %d\n", electron.h2oState);
//				}

                eAngles(&seed, &electron, elog, ei, e2nd, eRatio, &polar, &azi, &polar_e2nd, &azi_e2nd, elastDCSTable);
                eHop(&seed, &electron, polar, azi);
                eTime(&electron, ei);
                e2ndQ(&electron, container, e2Queue, e2nd, second_num, polar_e2nd, azi_e2nd, where, gEid);
			}
            eRecord(&electron, container, where, ContainerSize);
		}
		stepQ[i] = step;
		cntQ[i] = elascnt;//record the number of step and elastic step
#if ACTIVE_SEED == 1	
        if (MaxN > MAX_ACTIVE_THRD)
		{
		    states[seat] = seed;
		}
		else
		{
			states[i] = seed;
		}
#else
		states[i] = seed;
#endif
	}
}

void geteNum(int *e_num, char *infile)
{
	FILE *infilep = fopen(infile, "r");
	if(!infilep) 
	{
		fprintf(stderr, "Failed to open config file %s\n", infile);
		exit(EXIT_FAILURE);
	}

	fscanf(infilep, "%d\n", e_num);
	fclose(infilep);	
}

__global__ void setElectron(int num, float e, float RMAX, eStruct* d_eQueue, CURANDSTATE* cuseed)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    float r,cost,phi;
    if(id<num)
    {
        CURANDSTATE localstate=cuseed[id];
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

        d_eQueue[id].e = e;
        d_eQueue[id].h2oState = 99;
        d_eQueue[id].dead = 0;
        d_eQueue[id].path = 0;     
        d_eQueue[id].elape_time = 0.0;     
        d_eQueue[id].id = id;     
        d_eQueue[id].parentId = id;
        cuseed[id]=localstate;
    }   
}//*/
void iniElectron(float *max_e, eStruct *eQueue, int *id, CURANDSTATE* rnd_states, char *infile)
{
	FILE *infilep = fopen(infile, "r");
	if(!infilep) 
	{
		fprintf(stderr, "Failed to open config file %s\n", infile);
		exit(EXIT_FAILURE);
	}
    int num,flag;//
    float e,RMAX;
    fscanf(infilep,"%d %d\n",&num,&flag);
    fscanf(infilep,"%f\n",&e);
    fscanf(infilep,"%f\n",&RMAX);
    if(!flag)
    {
        eStruct* d_eQueue;
        cudaMalloc((void**)&d_eQueue,sizeof(eStruct)*num);
        setElectron<<<(num-1)/64+1,64>>>(num,e,RMAX,d_eQueue,rnd_states);
        cudaDeviceSynchronize();
        cudaMemcpy(eQueue,d_eQueue,sizeof(eStruct)*num,cudaMemcpyDeviceToHost);
        cudaFree(d_eQueue);
        *max_e=e;
        (*id)+=num;
    }
    else
    {
        float inx, iny, inz, t, dirx, diry, dirz, e;   
        printf("First (5) incident electrons : \n");
        for(int i = 0; i < num; i++)
        {
            fscanf(infilep, "%f %f %f %f %f %f %f %f\n", &inx, &iny, &inz, &t, &dirx, &diry, &dirz, &e);
            eQueue[i].x = inx;
            eQueue[i].y = iny;
            eQueue[i].z = inz;        
            eQueue[i].ux = dirx;
            eQueue[i].uy = diry;
            eQueue[i].uz = dirz;        
            eQueue[i].e = e;
            eQueue[i].h2oState = 99;
            eQueue[i].dead = 0;
            eQueue[i].path = 0;     
            eQueue[i].elape_time = t;     
            eQueue[i].id = i;     
            eQueue[i].parentId = i;
            (*id)++;
            if (e > *max_e)
                *max_e = e;
            if (i < 5)
                printf("(%f, %f, %f, %e)\n", inx, iny, inz, e);
        }
    }	
	fclose(infilep);	
}

int cmp( const void *a ,const void *b)
{
    return ((eStruct *)a)->elape_time > ((eStruct *)b)->elape_time ? 1 : -1;
}

int cmpcontainer( const void *a ,const void *b)
{
    return ((data *)a)->time > ((data *)b)->time ? 1 : -1;
}

int getSPcores(cudaDeviceProp devProp)
{  
    int cores = 0;
    int mp = devProp.multiProcessorCount;
    switch (devProp.major){
     case 2: // Fermi
      if (devProp.minor == 1) cores = mp * 48;
      else cores = mp * 32;
      break;
     case 3: // Kepler
      cores = mp * 192;
      break;
     case 5: // Maxwell
      cores = mp * 128;
      break;
     case 6: // Pascal
      if (devProp.minor == 1) cores = mp * 128;
      else if (devProp.minor == 0) cores = mp * 64;
      else printf("Unknown device type\n");
      break;
     default:
      printf("Unknown device type\n"); 
      break;
      }
    return cores;
}

void readConfigFile(const char *filename, REAL *Ecutoff, REAL *Idle, char *infile, int *GPUDeviceNo)
{

	FILE *configFile = fopen(filename, "r");
	char temp[200];

	if(!configFile) 
	{
		fprintf(stderr, "Failed to open config file %s\n", filename);
		exit(EXIT_FAILURE);
	}

	fgets(temp, 200, configFile);
	fscanf(configFile, "%d\n", GPUDeviceNo);	
	
	fgets(temp, 200, configFile);
	fscanf(configFile, "%f\n", Ecutoff);	
	
	fgets(temp, 200, configFile);
	fscanf(configFile, "%s\n", infile);	
	
	fgets(temp, 200, configFile);   
	fscanf(configFile, "%f\n", Idle);
	fclose(configFile);

}


int main(int argc, char *argv[]) 
{
	REAL **DACSTable = NULL;
	REAL *BindE_array = NULL;
	REAL **elastDCSTable = NULL;
	REAL **ieeCSTable = NULL;
	int i;
	time_t start_time,end_time; /* calendar time */
    start_time=clock(); /* get current cal time */
//	printf("  %s",asctime( localtime(&ltime) ) );
	// index  11 : DACS           10 : Elastic       -1 : Cutoff
	//       5-9 : Excitation    0-4 : Ionization
	
    long long where_all = 0;
    long long where = 0;
	int second_num = 0;
    int e_num;
	unsigned int state[13] = {0};
	int gEid = 0;
	int currentID=0;
    time_t t;
	srand((unsigned)time(&t));
	float acc_kerneltime = 0;
	
	int *dev_where, *dev_second_num, *dev_gEid;
	eStruct *dev_eQueue, *dev_e2Queue;
	int *dev_stepQ,  *dev_cntQ;
	data *dev_container;
	CURANDSTATE* rnd_states;

	int GPUDeviceNo;
	char *argvBuffer = argv[1];
	float idle;
	char infile[200];
	REAL Ecutoff;
	if (argc != 2)
    {
        printf("Please execute ./microMC config_file\n");
        printf("Thanks.\n\n");
        exit(1);
    }	
    readConfigFile(argvBuffer, &Ecutoff, &idle, infile, &GPUDeviceNo);
    printf("Cutoff energy = %.2f eV\n", Ecutoff);
	cudaSetDevice(GPUDeviceNo);
	cudaDeviceReset();
    size_t total_memory;
    size_t free_memory;
	cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, GPUDeviceNo);
    //int cudacores = getSPcores(deviceProp);
	//printf("cuda core = %d\n", cudacores);
	int globalMB = deviceProp.totalGlobalMem >> 20;
	printf("Total global memory = %d (MB)\n", globalMB);
	size_t sharedB = deviceProp.sharedMemPerBlock;
	printf("Shared memory per block = %zd (Bytes)\n", sharedB);
	
	
	// 0-4 ionizaton 5-9 excitation 10 elastic
	ieeCSTable = (REAL**)malloc(sizeof(ieeCSTable)*(INTERACTION_TYPES-1));
	for (i=0; i<INTERACTION_TYPES-1; i++)
		 ieeCSTable[i]=(REAL*)malloc(sizeof(REAL)*E_ENTRIES);
	
	long int row0s = (long int)ieeCSTable[0];
	long int row1s = (long int)ieeCSTable[1];
	int ieecsPitch = (row1s - row0s)/sizeof(REAL);
	
	DACSTable = (REAL**)malloc(sizeof(DACSTable)*2);
	for (i=0; i<2; i++)
		 DACSTable[i]=(REAL*)malloc(sizeof(REAL)*DACS_ENTRIES);	
	 	 
	long int daRow0s = (long int)DACSTable[0];
	long int daRow1s = (long int)DACSTable[1];
	int dacsPitch = (daRow1s - daRow0s)/sizeof(REAL);
	
	elastDCSTable = (REAL**)malloc(sizeof(elastDCSTable)*ODDS);
	for (i=0; i<ODDS; i++)
		 elastDCSTable[i]=(REAL*)malloc(sizeof(REAL)*E_ENTRIES);
	 
	
	BindE_array = (REAL*)malloc(sizeof(REAL)*BINDING_ITEMS*2);	
	
	// read DACS table
	FILE *dacsFp = NULL;
	FILE *ioncsFp = NULL;
	FILE *elastCSfp = NULL;
	FILE *elastDCSfp = NULL;
	FILE *excitCSfp = NULL;
	

    rd_dacs(dacsFp, DACSTable);
    rd_ioncs(ioncsFp, BindE_array, ieeCSTable);
    rd_elast(elastCSfp, elastDCSfp, &ieeCSTable[10], elastDCSTable);
    rd_excit(excitCSfp, &BindE_array[5], &ieeCSTable[5]);

	/********************************/
    /****   < TEXTURE MEMORY >   ****/
	/********************************/
	
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	cudaArray *dev_DACSTable, *dev_BindE_array, *dev_ieeCSTable, *dev_elastDCSTable;
	// cuda memory
	cudaMallocArray(&dev_DACSTable, &channelDesc, dacsPitch, 2);
	cudaMallocArray(&dev_BindE_array, &channelDesc, BINDING_ITEMS*2, 1);
	cudaMallocArray(&dev_ieeCSTable, &channelDesc, ieecsPitch, (INTERACTION_TYPES-1));
	cudaMallocArray(&dev_elastDCSTable, &channelDesc, ieecsPitch, ODDS);

	// cuda memory
	cudaMemcpyToArray(dev_DACSTable    , 0, 0, &DACSTable[0][0], 2 * dacsPitch * sizeof(REAL), cudaMemcpyHostToDevice);
    //cudaMemcpy2DToArray(dev_DACSTable    , 0, 0, &DACSTable[0][0], 2 * dacsPitch * sizeof(REAL), dacsPitch * sizeof(REAL),2,cudaMemcpyHostToDevice);
	cudaMemcpyToArray(dev_BindE_array  , 0, 0, &BindE_array[0], 1 * BINDING_ITEMS*2 * sizeof(REAL), cudaMemcpyHostToDevice);
	cudaMemcpyToArray(dev_ieeCSTable   , 0, 0, &ieeCSTable[0][0], (INTERACTION_TYPES-1) * ieecsPitch * sizeof(REAL), cudaMemcpyHostToDevice);
	cudaMemcpyToArray(dev_elastDCSTable, 0, 0, &elastDCSTable[0][0], ODDS * ieecsPitch * sizeof(REAL), cudaMemcpyHostToDevice);

    // resource description -> from cuda memory
    struct cudaResourceDesc resD_DACSTable;
        memset(&resD_DACSTable, 0, sizeof(resD_DACSTable));
        resD_DACSTable.resType = cudaResourceTypeArray;
        resD_DACSTable.res.array.array = dev_DACSTable;    // array data comes from dev_DACSTable
    struct cudaResourceDesc resD_BindE_array;
        memset(&resD_BindE_array, 0, sizeof(resD_BindE_array));
        resD_BindE_array.resType = cudaResourceTypeArray;
        resD_BindE_array.res.array.array = dev_BindE_array;    // array data comes from dev_BindE_array
    struct cudaResourceDesc resD_ieeCSTable;
        memset(&resD_ieeCSTable, 0, sizeof(resD_ieeCSTable));
        resD_ieeCSTable.resType = cudaResourceTypeArray;
        resD_ieeCSTable.res.array.array = dev_ieeCSTable;    // array data comes from dev_ieeCSTable
    struct cudaResourceDesc resD_elastDCSTable;
                    memset(&resD_elastDCSTable, 0, 
					 sizeof(resD_elastDCSTable));
                            resD_elastDCSTable.resType = cudaResourceTypeArray;
                            resD_elastDCSTable.res.array.array 
						   = dev_elastDCSTable;

    // texture description -> from cuda memory
    struct cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0]   = cudaAddressModeClamp;
        texDesc.addressMode[1]   = cudaAddressModeClamp;
        texDesc.readMode         = cudaReadModeElementType;
        texDesc.filterMode       = cudaFilterModeLinear; //use cudaFilterModePoint if don't want data in table are filted.
		
        cudaTextureObject_t texObj_DACSTable    = 0,
		                    texObj_BindE_array  = 0,
		                    texObj_ieeCSTable   = 0,
		                    texObj_elastDCSTable = 0;

        cudaCreateTextureObject(&texObj_DACSTable    , &resD_DACSTable,   &texDesc, NULL);
        cudaCreateTextureObject(&texObj_BindE_array  , &resD_BindE_array, &texDesc, NULL);
        cudaCreateTextureObject(&texObj_ieeCSTable   , &resD_ieeCSTable,  &texDesc, NULL);
        cudaCreateTextureObject(&texObj_elastDCSTable, &resD_elastDCSTable, &texDesc, NULL);

    int run = 0;
    system("[ ! -d ./output ] && mkdir ./output"); // check existence of the folder
	/**********************************************/
    /****   < RANDOM NUMBER INITIALIZATION >   ****/
	/**********************************************/
    cudaMalloc((void**) &rnd_states, NPART * sizeof(CURANDSTATE));
    rand_init<<<(NPART+127)/128, 128>>>(time(0), rnd_states);

    float max_e = 0;
    
    geteNum(&e_num, infile);
    eStruct *eQueue_ini = (eStruct *)malloc(e_num * sizeof(eStruct));
    iniElectron(&max_e, eQueue_ini, &currentID, rnd_states, infile);    
/************************************/
    FILE* fp1=fopen("./output/totalsource.txt","a");
    for(int i = 0; i < e_num; i++)
    {
        fprintf(fp1, "%.9f %.9f %.9f %.8f %.8f %.8f\n", eQueue_ini[i].x,eQueue_ini[i].y,eQueue_ini[i].z,eQueue_ini[i].ux,eQueue_ini[i].uy,eQueue_ini[i].uz);
    }
    fclose(fp1);
    printf("finish writing source position\n");
/************************************/
    int memory_rqMB = GPUMEM_BASE_SIZE + e_num * GPUMEM_PER_INC;
    //printf("e_num = %d\n", e_num);
    printf("Estimated GPU memory usage = %d (MB)\n", memory_rqMB);
    //globalMB = 1000;
    int batch = 1 + memory_rqMB/globalMB;

    int MaxN;
    int ContainerSize;
    int enumPerBatch = e_num/batch;
    int scale = 1;
    if (max_e > 100e3)
        scale = 1 + max_e/100e3;

    ContainerSize = enumPerBatch * scale * INC_RADICALS_100KEV;
    MaxN = enumPerBatch * scale * INC_2NDPARTICLES_100KEV;
    long long contsize = e_num * scale * INC_RADICALS_100KEV;

    printf("Total incident particles = %d\n", e_num);
    printf("Simulation in batchs = %d\n", batch);
    printf("Estimated Max. total radical states = %lld\n", contsize);
    printf("Estimated Max. batch radical states = %d\n", ContainerSize);
    printf("Estimated Max. batch 2nd particles = %d\n", MaxN);
    
    
    cudaMemcpyToSymbol(Ecutoff_dev, &Ecutoff, sizeof(REAL), 0, cudaMemcpyHostToDevice);


    float *e2ndQueue_test = (float *)malloc(MaxN * sizeof(float));
    int *host_stepQ = (int *)malloc(MaxN * sizeof(int));
    int *host_cntQ = (int *)malloc(MaxN * sizeof(int));
    cudaMalloc(&dev_stepQ, MaxN * sizeof(int));
    cudaMalloc(&dev_cntQ, MaxN * sizeof(int));
    cudaMalloc(&dev_e2Queue, MaxN * sizeof(eStruct));
    cudaMalloc(&dev_container, ContainerSize * sizeof(data));
    cudaMalloc(&dev_where, sizeof(int));
    cudaMalloc(&dev_second_num, sizeof(int));
    cudaMalloc(&dev_gEid, sizeof(int));

    cudaMemcpy(dev_where, &where, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_second_num, &second_num, sizeof(int), cudaMemcpyHostToDevice);//*/

    cudaMalloc(&dev_eQueue, NPART * sizeof(eStruct));		
	cuMemGetInfo(&free_memory,&total_memory);
	printf("GPU memory usage = %zu (MB)\n\n", (total_memory-free_memory)>>20);
    
	int iRun = 0;
	data *h_container = (data *)malloc(contsize * sizeof(data));
    int sim_num = 0;
	
	for (i = 0; i<batch; i++)
	{
		if (i == batch - 1)//particle number per batch
		{
			sim_num = e_num-i*enumPerBatch;
		}
		else
		{
			sim_num = enumPerBatch;
		}
		
	    eStruct *eQueue = (eStruct *)malloc(sim_num * sizeof(eStruct));
	    memcpy(eQueue, &eQueue_ini[i*enumPerBatch], sim_num*sizeof(eStruct));
		
    	while (sim_num > 0)
    	{	
			
            iRun++;
    		printf("\nRunning particles : %d\n", sim_num);
            long long done_sim = 0; // number of particle simulated
			int cnt = 0;
			int cur_sim;
			
		    for (;;)
			{
				int N;
				if(done_sim == sim_num) break;
			    if (sim_num - done_sim >= NPART)
				{
					cur_sim = NPART;
    		        gEid+=cur_sim;
    	            cudaMemcpy(dev_gEid, &gEid, sizeof(int), cudaMemcpyHostToDevice);
    	            cudaMemcpy(dev_second_num, &second_num, sizeof(int), cudaMemcpyHostToDevice);
    	            cudaMemcpy(dev_eQueue, &eQueue[cnt*NPART], NPART * sizeof(eStruct), cudaMemcpyHostToDevice);					
					done_sim += cur_sim;
    		        N = cur_sim;
				}
				else
				{
					cur_sim = sim_num-done_sim;
    		        gEid+=cur_sim;
    	            cudaMemcpy(dev_gEid, &gEid, sizeof(int), cudaMemcpyHostToDevice);
    	            cudaMemcpy(dev_second_num, &second_num, sizeof(int), cudaMemcpyHostToDevice);
    	            cudaMemcpy(dev_eQueue, &eQueue[cnt*NPART], cur_sim * sizeof(eStruct), cudaMemcpyHostToDevice);					
					done_sim = sim_num;
    		        N = cur_sim;
				}
				
                cudaEvent_t start, stop;
                cudaEventCreate(&start);
                cudaEventCreate(&stop);
                float milliseconds = 0.0;
                cudaEventRecord(start, 0);
    		    eTransport<<<(N+K_THREADS-1)/K_THREADS, K_THREADS>>>(N, ContainerSize, dev_eQueue, dev_e2Queue, dev_stepQ, dev_cntQ, dev_container, rnd_states, dev_where, dev_second_num, dev_gEid,
                                             texObj_DACSTable, texObj_BindE_array, texObj_ieeCSTable, texObj_elastDCSTable, iRun, MaxN);
                cudaEventRecord(stop, 0);
		        cudaEventSynchronize(stop);
		        cudaEventElapsedTime(&milliseconds, start, stop);
		        printf(">>>> Kernel time = %f (ms)\n", milliseconds);
				acc_kerneltime += milliseconds;
		        cudaCHECK();
	            cudaMemcpy(&second_num, dev_second_num, sizeof(int), cudaMemcpyDeviceToHost);

                cnt++;
			}   
    		
    		free(eQueue);
	        cudaMemcpy(&second_num, dev_second_num, sizeof(int), cudaMemcpyDeviceToHost);
			if (second_num > MaxN) 
			{
				printf("Error : Generated second particles exceed Max value.\n");
				exit(1);
			}    		
    #if QSORT == 1
     		printf("sorting by thrust\n");
  		    thrust::device_ptr<eStruct> dev_ptr(dev_e2Queue);
			
            compare_eStruct sort_cmp;
    		thrust::sort(dev_ptr, dev_ptr+second_num, sort_cmp);
			
    #endif
    		
    //		printf("run = %d second_num = %d sim_num = %d\n", run, second_num, sim_num);
	        int pre_num = sim_num;
    		sim_num = second_num;
    		second_num = 0;
    	    eQueue = (eStruct *)malloc(sim_num * sizeof(eStruct));
    		cudaMemcpy(eQueue, dev_e2Queue, sim_num * sizeof(eStruct), cudaMemcpyDeviceToHost);
    #if QSORT == 2
     		printf("sorting by qsort\n");
    		qsort(eQueue, sim_num, sizeof(eStruct), cmp);
    #endif			
    		/*cudaMemcpy(host_stepQ, dev_stepQ, pre_num * sizeof(int), cudaMemcpyDeviceToHost);
    		cudaMemcpy(host_cntQ, dev_cntQ, pre_num * sizeof(int), cudaMemcpyDeviceToHost);
			//printf(">>>> Steps\n");
			int stepLoc, maxStep = 0;
	        for (int w=0; w<pre_num; w++)
            {
				if (host_stepQ[w] > maxStep)
				{
					maxStep = host_stepQ[w];
					stepLoc = w;
				}
			}				
			printf("maxStep = %d stepLoc = %d\n", maxStep, stepLoc);
			
			//printf(">>>> Elastic Steps\n");
	        //for (int w=0; w<pre_num; w++) printf("%d\n", host_cntQ[w]);
			printf("elastep = %d\n", host_cntQ[stepLoc]);
			
			//printf(">>>> Energy\n");
			float max2ndE = 0.0;
	        for (int w=0; w<sim_num; w++)
			{
				if (eQueue[w].e > max2ndE)
				{
					max2ndE = eQueue[w].e;
				}
				e2ndQueue_test[w] = eQueue[w].e;
			}
            printf("maxStepE = %f\n", e2ndQueue_test[stepLoc]);
			printf("e2ndMax = %f\n", max2ndE);//*/
    	    cudaMemcpy(&where, dev_where, sizeof(int), cudaMemcpyDeviceToHost);
//            printf("where=%d\n", where);
            run++;
   	    }
	    cudaMemcpy(&where, dev_where, sizeof(int), cudaMemcpyDeviceToHost);
	    cudaMemcpy(&h_container[where_all], dev_container, where * sizeof(data), cudaMemcpyDeviceToHost);
		where_all += where;
		where = 0;
	    cudaMemcpy(dev_where, &where, sizeof(int), cudaMemcpyHostToDevice);		
    }
	
    printf("\n\nGPU Kernel time = %f (ms)\n", acc_kerneltime);

    printf("\n\nRecorded total states=%lld\n", where_all);

#if QSORT==3	
    qsort(h_container,where_all,sizeof(data),cmpcontainer);
#endif 
    float x,y,z,r2,e;
	//for bin output test 6th Oct
    float deposit_e = 0.0,total_e=0.0,ecutoff=0;
    FILE* totalphy=fopen("./output/totalphy.dat","ab");   
    FILE* physint=fopen("./output/phyint.dat","wb");
    FILE* physfloat=fopen("./output/phyfloat.dat","wb");
    int outphyint=0;
    
	
#if REPORT_FILE == 1   
	for (long long i=0; i<where_all; i++)
	{
		x = 1e7*h_container[i].x; // cm to nm
        y = 1e7*h_container[i].y;
        z = 1e7*h_container[i].z;
        r2 = x*x+y*y+z*z;
        //if(r2>6700*6700) continue;
        fwrite(&(h_container[i].parentId),sizeof(int),1,physint);
        fwrite(&(h_container[i].id),sizeof(int),1,physint);
        
        if (h_container[i].h2oState == -1)
        {
            outphyint=0;
            state[12]++;
        }
        else
        {
            outphyint=7;
            state[h_container[i].h2oState]++;
        }
        fwrite(&outphyint,sizeof(int),1,physint);
        
        if (h_container[i].h2oState == 11)   // DA changes to 10 for fitting prechem.
            outphyint=10;
        else
            outphyint=h_container[i].h2oState;
        fwrite(&outphyint,sizeof(int),1,physint);

		if (h_container[i].h2oState == -1) ecutoff+= h_container[i].e;
        if (h_container[i].h2oState == -1 || h_container[i].h2oState == 11)
		{
            total_e+= h_container[i].e;
            
            if(r2<NUCLEUS_RADIUS*NUCLEUS_RADIUS) 
            {
                deposit_e += h_container[i].e;
                fwrite (&x, sizeof(float), 1, totalphy );
                fwrite (&y, sizeof(float), 1, totalphy );
                fwrite (&z, sizeof(float), 1, totalphy );
                fwrite (&(h_container[i].e), sizeof(float), 1, totalphy );
            }
		    fwrite(&(h_container[i].e),sizeof(float),1,physfloat);          
		}
		else
		{
            total_e+=BindE_array[h_container[i].h2oState];
            if(r2<NUCLEUS_RADIUS*NUCLEUS_RADIUS) 
            {
                deposit_e += BindE_array[h_container[i].h2oState];
                fwrite (&x, sizeof(float), 1, totalphy );
                fwrite (&y, sizeof(float), 1, totalphy );
                fwrite (&z, sizeof(float), 1, totalphy );
                fwrite (&BindE_array[h_container[i].h2oState], sizeof(float), 1, totalphy );
            }
            fwrite(&(BindE_array[h_container[i].h2oState]),sizeof(float),1,physfloat);
            
		}	
        fwrite(&(x),sizeof(float),1,physfloat);         
        fwrite(&(y),sizeof(float),1,physfloat);       
        fwrite(&(z),sizeof(float),1,physfloat);        
        fwrite(&(h_container[i].time),sizeof(float),1,physfloat);
	}
#endif
    outphyint=state[12];
    fwrite(&outphyint,sizeof(int),1,physint);
    outphyint=state[0]+state[1]+state[2]+state[3]+state[4];
    fwrite(&outphyint,sizeof(int),1,physint);
    outphyint=state[5];
    fwrite(&outphyint,sizeof(int),1,physint);
    outphyint=state[6];
    fwrite(&outphyint,sizeof(int),1,physint);
    outphyint=state[7]+state[8]+state[9];
    fwrite(&outphyint,sizeof(int),1,physint);
    outphyint=state[11];
    fwrite(&outphyint,sizeof(int),1,physint);


    fclose(totalphy);
    fclose(physint);
    fclose(physfloat);
    printf("total deposited energy= %f eV\n", total_e);
    printf("total deposited cutoff energy= %f eV\n", ecutoff);
    printf("energy deposit inside the nucleus = %f eV\n", deposit_e);

	
#if E_DEPOSIT_FILE == 1
    float dep_sum=0,dep_total=0;
	FILE *depofp = NULL;
	char filename[] = "./output/deposit.txt";
    depofp = fopen(filename, "r");
	if (depofp == NULL)
	{
		printf("The file 'deposit.txt' doesn't exist\n");
        printf("Deposited energy will be initialized to 0 0 \n");
		dep_sum = 0;
        dep_total = 0;
	}
    else
    {
        fscanf(depofp, "%f %f", &dep_sum,&dep_total);
        fclose(depofp);
    }
	
	dep_sum += deposit_e;
    dep_total += total_e;
    depofp = fopen("./output/deposit.txt", "w");
    fprintf(depofp, "%f %f", dep_sum,dep_total);
	fclose(depofp);
#endif

	cudaFree(dev_e2Queue);
	cudaFree(dev_container);
	cudaFree(dev_where);
	cudaFree(dev_second_num);
	cudaFree(rnd_states);
	cudaFree(dev_DACSTable);
	cudaFree(dev_BindE_array);
	cudaFree(dev_ieeCSTable);
	cudaFree(dev_elastDCSTable);
	
	
	free(DACSTable);
	free(BindE_array);
	free(elastDCSTable);
	free(ieeCSTable);
	
//	free(eQueue);
//	free(e2Queue);
	free(h_container);
    end_time=clock();
    printf("Total computation time: %f seconds.\n\n", ((float)end_time-(float)start_time)/CLOCKS_PER_SEC);
}
