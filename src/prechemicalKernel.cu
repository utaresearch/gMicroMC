#include "prechemicalKernel.cuh"
#include "prechemical.h"
float *d_posx, *d_posy, *d_posz; // the GPU variables to store the positions of the particles (a larger memory is required to include the product of prechemical stage) 
float *d_ene, *d_ttime; // initial energies of the initial particles
int *d_ptype, *d_index; // the species type of the particles (255 for empty entries or produced H2O)	
int *d_statetype; // the statetype of the initial particles
int *d_wiid_elec;// the parent ion id of electrons for potential recombination

cudaArray *d_p_recomb_elec;
cudaArray *d_rms_therm_elec; 
texture<float,1,cudaReadModeElementType> p_recomb_elec_tex;	
texture<float,1,cudaReadModeElementType> rms_therm_elec_tex;	

__device__ __constant__ int d_num_elec; 
__device__ __constant__ int d_num_wi; 
__device__ __constant__ int d_num_we_a1b1; 
__device__ __constant__ int d_num_we_b1a1; 
__device__ __constant__ int d_num_we_rd; 
__device__ __constant__ int d_num_w_dis; 
__device__ __constant__ int d_num_total; 
	
__device__ __constant__ int d_sidx_elec; 
__device__ __constant__ int d_sidx_wi; 
__device__ __constant__ int d_sidx_we_a1b1; 
__device__ __constant__ int d_sidx_we_b1a1; 
__device__ __constant__ int d_sidx_we_rd; 
__device__ __constant__ int d_sidx_w_dis; 

__device__ __constant__ int d_nebin;
__device__ __constant__ float d_mine_ebin;
__device__ __constant__ float d_ide_ebin;

__device__ __constant__ int d_nbtype;
__device__ float d_placeinfo_btype[MAXBRANCHTYPE*9];
__device__ int d_num_product_btype[MAXBRANCHTYPE];
__device__ int d_ptype_product_btype[MAXBRANCHTYPE * MAXNUMBRANCHPROD];

__device__ __constant__ int d_nb_rece;
__device__ __constant__ int d_nb_wi; 
__device__ __constant__ int d_nb_we_a1b1; 
__device__ __constant__ int d_nb_we_b1a1; 
__device__ __constant__ int d_nb_we_rd; 
__device__ __constant__ int d_nb_w_dis; 
	
__device__ float d_pb_rece[MAXNUMBRANCH];
__device__ float d_pb_wi[MAXNUMBRANCH]; 
__device__ float d_pb_we_a1b1[MAXNUMBRANCH]; 
__device__ float d_pb_we_b1a1[MAXNUMBRANCH];
__device__ float d_pb_we_rd[MAXNUMBRANCH];
__device__ float d_pb_w_dis[MAXNUMBRANCH];

__device__ int d_btype_rece[MAXNUMBRANCH];
__device__ int d_btype_wi[MAXNUMBRANCH]; 
__device__ int d_btype_we_a1b1[MAXNUMBRANCH]; 
__device__ int d_btype_we_b1a1[MAXNUMBRANCH];
__device__ int d_btype_we_rd[MAXNUMBRANCH];
__device__ int d_btype_w_dis[MAXNUMBRANCH];

__global__ void thermalisation_subexelectrons(float *d_posx, // x position of the particles (input and output)
                                              float *d_posy,
											  float *d_posz,
											  float *d_ene, // initial energies of the initial particles (input only)
											  int *d_ptype, // species type for products of prechemical stage, 255 for empty or produced water (output)
											  int *d_statetype, //the statetype of the initial particles (255 for died particles)
											  int *d_wiid_elec) // the index of the ionized water molecule for potential recombination
{											 
    const int tid = blockIdx.x*blockDim.x+ threadIdx.x;
	const int pid = tid + d_sidx_elec;
    
	if(tid < d_num_elec && d_statetype[pid] < 255)
	{
	     curandState localState = cuseed[pid];
		 
		 float radnum = curand_uniform(&localState);
		 
		 float idx_ebin = d_ide_ebin*(d_ene[pid] - d_mine_ebin) + 0.5;	 
		 
		 float temp = tex1D(p_recomb_elec_tex, idx_ebin);
	     //float temp = 0.26f;
		 
         //printf("thermalisation_subexelectrons: pid = %d, temp = %f, radnum = %f\n", pid, temp, radnum);	
		if (d_ene[pid]>7.5)
			{printf("d_ene=%f, indx_ebin=%f,temp=%f\n", d_ene[pid], idx_ebin,temp);}
		if(radnum < temp) 
		{
		  // the subexcitation electron recombine with its parent ion to form H2O* first, then H2O* has two deexcitation branches in our simulation as follows
		  // 1) dissociative deexcitation branch to become either OH. + H. (PBRANCH11RECOMB 55%) or H2 + OH.+ OH. (PBRANCH12RECOMB 15%)
		  // here we directly simulate the deexcitation of the recombined H2O*
		  // 2) nondissociative deexcitation branch to become H2O + energy (PBRANCH2RECOMB 30%)
		  int wiid = d_wiid_elec[tid];
		  
		  d_statetype[pid] = 255;
		  d_statetype[wiid] = 255;
	  
		  radnum = curand_uniform(&localState);
		  if(radnum < PBRANCH11RECOMB)
		  {
		    // OH. + H.
		    int btype = 1;
			
			d_ptype[pid] = d_ptype_product_btype[btype * MAXNUMBRANCHPROD]; //OH.
			d_ptype[pid + d_num_total] = d_ptype_product_btype[btype * MAXNUMBRANCHPROD + 1]; //H.
			
			displace_twoproducts_noholehoping(d_posx, d_posy, d_posz, &localState, btype, pid, wiid); // note that we consider the site of the recombination happened at the location of the parent ionized water                                              										
		  }
		  else if(radnum < PBRANCH11RECOMB + PBRANCH12RECOMB)
		  {
		    // H2 + OH.+ OH.
			int btype = 4;
			
			d_ptype[pid] = d_ptype_product_btype[btype * MAXNUMBRANCHPROD]; //H2
			d_ptype[pid + d_num_total] = d_ptype_product_btype[btype * MAXNUMBRANCHPROD + 1]; //OH.
			d_ptype[pid + 2 * d_num_total] = d_ptype_product_btype[btype * MAXNUMBRANCHPROD + 2]; //OH. (written into the empty entry)
						
			displace_threeproducts_noholehoping(d_posx, d_posy, d_posz, &localState, btype, pid, wiid);	// note that we consider the site of the recombination happened at the location of the parent ionized water 		
		  }
		  // no performance needed for the third channel which only produce H2O
		  else
		  	{int btype=2;}
		}
		else
		{
		  // thermalization 
		  d_statetype[pid] = 255;
		  d_ptype[pid] = 0;
		  
		  float ndisx, ndisy, ndisz;
		  sampleThermalDistance(pid, &localState, &ndisx, &ndisy, &ndisz, idx_ebin);
		  		  
		  d_posx[pid] += ndisx;
		  d_posy[pid] += ndisy;
		  d_posz[pid] += ndisz;
		}
		
		cuseed[pid] = localState;
	}
	
}											  



__global__ void dissociation_ionizedwater(float *d_posx,
                                          float *d_posy,
										  float *d_posz,
										  int *d_ptype,
										  int *d_statetype)										  
{
    const int tid = blockIdx.x*blockDim.x+ threadIdx.x;
	const int pid = tid + d_sidx_wi;
	
	if(tid < d_num_wi && d_statetype[pid] < 255)
	{  
	// only one branch for ionized water , H3O+, OH.
	    curandState localState = cuseed[pid];

	    d_statetype[pid] = 255;
		   
	    int btype = d_btype_wi[0];
		
		d_ptype[pid] = d_ptype_product_btype[btype * MAXNUMBRANCHPROD];			
		d_ptype[pid + d_num_total] = d_ptype_product_btype[btype * MAXNUMBRANCHPROD + 1];
		
		displace_twoproducts_holehoping(d_posx, d_posy, d_posz, &localState, btype, pid);   
		
        cuseed[pid] = localState;		
	}
}

__global__ void dissociation_excitedwater_a1b1(float *d_posx,
                                               float *d_posy,
										       float *d_posz,
										       int *d_ptype,
										       int *d_statetype)										  
{
    const int tid = blockIdx.x*blockDim.x+ threadIdx.x;
	const int pid = tid + d_sidx_we_a1b1;
	
	if(tid < d_num_we_a1b1 && d_statetype[pid] < 255)
	{  
	// two branches for a1b1 excited water 
	    curandState localState = cuseed[pid];

	    d_statetype[pid] = 255;
		
		float radnum = curand_uniform(&localState);
		int btype;
		
		
		if(radnum < d_pb_we_a1b1[0])
		{
		    btype = d_btype_we_a1b1[0];
		}
		else
		{
		    btype = d_btype_we_a1b1[1];
		}
		   
        //printf("dissociation_excitedwater_a1b1: pid = %d, radnum = %f, btype = %d\n", pid, radnum, btype);
		
		if(d_num_product_btype[btype] > 0)
		{
			d_ptype[pid] = d_ptype_product_btype[btype * MAXNUMBRANCHPROD];			
		    d_ptype[pid + d_num_total] = d_ptype_product_btype[btype * MAXNUMBRANCHPROD + 1];
			
			displace_twoproducts_noholehoping(d_posx, d_posy, d_posz, &localState, btype, pid, pid);       
		}	
        // for the branch channel to produce H2O only, nothing needs to be performed
        		
		cuseed[pid] = localState;
	}
}

__global__ void dissociation_excitedwater_b1a1(float *d_posx,
                                               float *d_posy,
										       float *d_posz,
										       int *d_ptype,
										       int *d_statetype)										  
{
    const int tid = blockIdx.x*blockDim.x+ threadIdx.x;
	const int pid = tid + d_sidx_we_b1a1;
	
	if(tid < d_num_we_b1a1 && d_statetype[pid] < 255)
	{  
	// three branches for b1a1 excited water 
	    curandState localState = cuseed[pid];

	    d_statetype[pid] = 255;
		
		float radnum = curand_uniform(&localState);
		int btype;
		
		if(radnum < d_pb_we_b1a1[0])
		{
		    btype = d_btype_we_b1a1[0];
		}
		else if(radnum < d_pb_we_b1a1[0]+d_pb_we_b1a1[1])
		{
		    btype = d_btype_we_b1a1[1];
		}
		else
		{
		    btype = d_btype_we_b1a1[2];  
		}
		
        //printf("dissociation_excitedwater_b1a1: pid = %d, radnum = %f, btype = %d\n", pid, radnum, btype);
		
		if(d_num_product_btype[btype] >0 )
		{
			d_ptype[pid] = d_ptype_product_btype[btype * MAXNUMBRANCHPROD];			
			d_ptype[pid + d_num_total] = d_ptype_product_btype[btype * MAXNUMBRANCHPROD + 1];
			d_ptype[pid + d_num_total * 2] = d_ptype_product_btype[btype * MAXNUMBRANCHPROD + 2];
			
			if(d_placeinfo_btype[btype * 9 + 7] == 0 && d_placeinfo_btype[btype * 9 + 8] == 0)
			{
			   // the branch channel H3O+, OH., eaq(0.025 eV)			    			
				displace_twoproducts_oneelec_holehoping(d_posx, d_posy, d_posz, &localState, btype,pid);                                                       															
			}
			else
			{
			   // the branch channel OH., OH., H2
			   displace_threeproducts_noholehoping(d_posx, d_posy, d_posz, &localState, btype, pid, pid);				
			}			
		}	
		// for the branch channel to produce H2O only, nothing needs to be performed
		
		cuseed[pid] = localState;
	}
}


__global__ void dissociation_excitedwater_rd(float *d_posx,
                                             float *d_posy,
										     float *d_posz,
										     int *d_ptype,
										     int *d_statetype)										  
{
    const int tid = blockIdx.x*blockDim.x+ threadIdx.x;
	const int pid = tid + d_sidx_we_rd;
	
	if(tid < d_num_we_rd && d_statetype[pid] < 255)
	{  
	// two branches for rydberg and diffusion bands excited water 
	    curandState localState = cuseed[pid];

	    d_statetype[pid] = 255;
		
		float radnum = curand_uniform(&localState);
		int btype;
		
		if(radnum < d_pb_we_rd[0])
		{
		    btype = d_btype_we_rd[0];
		}
		else
		{
		    btype = d_btype_we_rd[1];
		}
		
        //printf("dissociation_excitedwater_rd: pid = %d, radnum = %f, btype = %d\n", pid, radnum, btype);
		
		if(d_num_product_btype[btype] >0 )
		{
			// the branch channel H3O+, OH., eaq
			d_ptype[pid] = d_ptype_product_btype[btype * MAXNUMBRANCHPROD];			
			d_ptype[pid + d_num_total] = d_ptype_product_btype[btype * MAXNUMBRANCHPROD + 1];
			d_ptype[pid + d_num_total * 2] = d_ptype_product_btype[btype * MAXNUMBRANCHPROD + 2];
			
			displace_twoproducts_oneelec_holehoping(d_posx, d_posy, d_posz, &localState, btype,pid);                   
		}	
		// for the branch channel to produce H2O only, nothing needs to be performed
		
		cuseed[pid] = localState;
	}
}

__global__ void dissociation_dissociativewater(float *d_posx,
                                               float *d_posy,
										       float *d_posz,
										       int *d_ptype,
										       int *d_statetype)										  
{
    const int tid = blockIdx.x*blockDim.x+ threadIdx.x;
	const int pid = tid + d_sidx_w_dis;
	
	if(tid < d_num_w_dis && d_statetype[pid] < 255)
	{  
	// one branches for dissociative water 
	    curandState localState = cuseed[pid];

	    d_statetype[pid] = 255;
			
		int btype = d_btype_w_dis[0];
		
        //printf("dissociative water: pid = %d, btype = %d\n", pid, btype);
		
		// the branch channel H2, OH., OH-
		d_ptype[pid] = d_ptype_product_btype[btype * MAXNUMBRANCHPROD];			
		d_ptype[pid + d_num_total] = d_ptype_product_btype[btype * MAXNUMBRANCHPROD + 1];
		d_ptype[pid + d_num_total * 2] = d_ptype_product_btype[btype * MAXNUMBRANCHPROD + 2];
		
        displace_threeproducts_noholehoping(d_posx, d_posy, d_posz, &localState, btype, pid, pid);       		
		
		cuseed[pid] = localState;
	}
}

__device__ void displace_twoproducts_noholehoping(float *d_posx, 
                                                  float *d_posy, 
												  float *d_posz,
												  curandState *localState_pt,
												  int btype, //branch type
												  int pid, // the current particle id
												  int pid_site) // the id of the particle considerred to be the original site (for recombination)
{
    
	float posx_site = d_posx[pid_site]; 
	float posy_site = d_posy[pid_site]; 
	float posz_site = d_posz[pid_site]; 
	
	float std_dis = d_placeinfo_btype[btype * 9 + 1]/__fsqrt_rn(3.0f);
	
	float coef = d_placeinfo_btype[btype * 9 + 3];
    
	float ndisx, ndisy, ndisz;
	getNormalizedDis_Sample3DGuassian(localState_pt, &ndisx, &ndisy, &ndisz);
	
	d_posx[pid] = posx_site + std_dis * ndisx * coef;
	d_posy[pid] = posy_site + std_dis * ndisy * coef;
	d_posz[pid] = posz_site + std_dis * ndisz * coef;
	
	coef = d_placeinfo_btype[btype * 9 + 5];
	d_posx[pid + d_num_total] = posx_site + std_dis * ndisx * coef;
	d_posy[pid + d_num_total] = posy_site + std_dis * ndisy * coef;
	d_posz[pid + d_num_total] = posz_site + std_dis * ndisz * coef;
	
	//printf("displace_twoproducts_noholehoping: pid = %d, pid_site = %d, std_dis = %f, ndisx = %f, ndisy = %f, ndisz = %f\n", pid, pid_site, std_dis, ndisx, ndisy, ndisz);
}

__device__ void displace_threeproducts_noholehoping(float *d_posx, 
                                                  float *d_posy, 
												  float *d_posz, 
												  curandState *localState_pt,
												  int btype, //branch type
												  int pid, // the current particle id
												  int pid_site) // the id of the particle considerred to be the original site (for recombination)
{
    			
	float posx_site = d_posx[pid_site]; 
	float posy_site = d_posy[pid_site]; 
	float posz_site = d_posz[pid_site]; 
	
	float std_dis = d_placeinfo_btype[btype * 9 + 1]/__fsqrt_rn(3.0f);
	float ndisx, ndisy, ndisz;
	getNormalizedDis_Sample3DGuassian(localState_pt, &ndisx, &ndisy, &ndisz);
	
	float coef = d_placeinfo_btype[btype * 9 + 3];
	d_posx[pid] = posx_site + std_dis * ndisx * coef;
	d_posy[pid] = posy_site + std_dis * ndisy * coef;
	d_posz[pid] = posz_site + std_dis * ndisz * coef;
	
	coef = d_placeinfo_btype[btype * 9 + 5];
	d_posx[pid + d_num_total] = posx_site + std_dis * ndisx * coef;
	d_posy[pid + d_num_total] = posy_site + std_dis * ndisy * coef;
	d_posz[pid + d_num_total] = posz_site + std_dis * ndisz * coef;
	
	coef = d_placeinfo_btype[btype * 9 + 7];
	d_posx[pid + 2 * d_num_total] = posx_site + std_dis * ndisx * coef;
	d_posy[pid + 2 * d_num_total] = posy_site + std_dis * ndisy * coef;
	d_posz[pid + 2 * d_num_total] = posz_site + std_dis * ndisz * coef;
	
	//printf("displace_threeproducts_noholehoping-1: pid = %d, pid_site = %d, std_dis = %f, ndisx = %f, ndisy = %f, ndisz = %f\n", pid, pid_site, std_dis, ndisx, ndisy, ndisz);
	
	std_dis = d_placeinfo_btype[btype * 9 + 2]/__fsqrt_rn(3.0f);
	getNormalizedDis_Sample3DGuassian(localState_pt, &ndisx, &ndisy, &ndisz);
	
	coef = d_placeinfo_btype[btype * 9 + 6];
	d_posx[pid + d_num_total]+= std_dis * ndisx * coef;
	d_posy[pid + d_num_total]+= std_dis * ndisy * coef;
	d_posz[pid + d_num_total]+= std_dis * ndisz * coef;
	
	coef = d_placeinfo_btype[btype * 9 + 8];
	d_posx[pid + 2 * d_num_total]+= std_dis * ndisx * coef;
	d_posy[pid + 2 * d_num_total]+= std_dis * ndisy * coef;
	d_posz[pid + 2 * d_num_total]+= std_dis * ndisz * coef;	
	
	//printf("displace_threeproducts_noholehoping-2: pid = %d, std_dis = %f, ndisx = %f, ndisy = %f, ndisz = %f\n", pid, std_dis, ndisx, ndisy, ndisz);
}

__device__ void displace_twoproducts_holehoping(float *d_posx, 
                                                float *d_posy, 
												float *d_posz, 
												curandState *localState_pt,
												int btype, //branch type
												int pid) // the current particle id											  
{
    float ndisx_hole, ndisy_hole, ndisz_hole, std_hole, ndisx, ndisy, ndisz, std_dis;	
	std_hole = d_placeinfo_btype[btype*9]/__fsqrt_rn(3.0f);				 	
	getNormalizedDis_Sample3DGuassian(localState_pt, &ndisx_hole, &ndisy_hole, &ndisz_hole);				   
						   
	float posx_site = d_posx[pid] + ndisx_hole * std_hole; 
	float posy_site = d_posy[pid] + ndisy_hole * std_hole; 
	float posz_site = d_posz[pid] + ndisz_hole * std_hole; 
		   
	std_dis = d_placeinfo_btype[btype*9 + 1]/__fsqrt_rn(3.0f);
	getNormalizedDis_Sample3DGuassian(localState_pt, &ndisx, &ndisy, &ndisz);	
		   
	float coef = d_placeinfo_btype[btype*9 + 3];   
	float radnum = curand_uniform(localState_pt);
	//printf("displace_twoproducts_holehoping: pid = %d, radnum = %f, ndisx_hole = %f, ndisy_hole = %f, ndisz_hole = %f, std_hole = %f, ndisx = %f, ndisy = %f, ndisz = %f, std_dis = %f\n", pid, radnum, ndisx_hole, ndisy_hole, ndisz_hole, std_hole, ndisx, ndisy, ndisz, std_dis);
	
	if(radnum < 0.5f)
	{
	  d_posx[pid] = posx_site;
	  d_posy[pid] = posy_site;
	  d_posz[pid] = posz_site;
	  
	  d_posx[pid + d_num_total] = posx_site + coef * ndisx * std_dis;
	  d_posy[pid + d_num_total] = posy_site + coef * ndisy * std_dis;
	  d_posz[pid + d_num_total] = posz_site + coef * ndisz * std_dis;
	}
	else
	{
	  d_posx[pid + d_num_total] = posx_site;
	  d_posy[pid + d_num_total] = posy_site;
	  d_posz[pid + d_num_total] = posz_site;
	  
	  d_posx[pid] = posx_site + coef * ndisx * std_dis;
	  d_posy[pid] = posy_site + coef * ndisy * std_dis;
	  d_posz[pid] = posz_site + coef * ndisz * std_dis;
	}	
}

__device__ void sampleThermalDistance(int pid, curandState *localState_pt, float *ndisx, float *ndisy, float *ndisz, float idx_ebin)
{
	float r  = curand_uniform(localState_pt) * 4.5f; // the probability for r>4.5 is 0.12%, ignore r>4.5f for efficiency purpose
	float radnum = curand_uniform(localState_pt) * 0.7357589;	// max value of 4r*exp(-2r), r=0.5 --> 2/e	  
	float temp = 4.0f*r*__expf(-2.0*r); // sampling function
	  
	while(radnum>temp)
	{
	    r  = curand_uniform(localState_pt) * 4.5f; 
		radnum = curand_uniform(localState_pt) * 0.7357589;
		temp = 4.0f*r*__expf(-2.0*r);			
	}
	  
	float rms_therm = tex1D(rms_therm_elec_tex, idx_ebin); //scaled by thermalization rms displacement		  
	getDirection_SampleOnSphereSurface(localState_pt, ndisx, ndisy, ndisz); 
	
//	printf("sampleThermalDistance: pid = %d, idx_ebin = %f, radnum = %f, temp = %f, r = %f, rms_therm = %f, ndisx = %f, ndisy = %f, ndisz = %f\n", pid, idx_ebin, radnum, temp, r, rms_therm, *ndisx, *ndisy, *ndisz);	
	*ndisx = (*ndisx) * r * rms_therm;
	*ndisy = (*ndisy) * r * rms_therm;
	*ndisz = (*ndisz) * r * rms_therm;	
}

__device__ void displace_twoproducts_oneelec_holehoping(float *d_posx, 
                                                        float *d_posy, 
												        float *d_posz, 
												        curandState *localState_pt,
												        int btype, //branch type
												        int pid) // the current particle id	
{
    //place the first two products with hole hoping, which is exactly same to function displace_twoproducts_holehoping();
	float ndisx_hole, ndisy_hole, ndisz_hole, std_hole, ndisx, ndisy, ndisz, std_dis;	
	std_hole = d_placeinfo_btype[btype*9]/__fsqrt_rn(3.0f);				 	
	getNormalizedDis_Sample3DGuassian(localState_pt, &ndisx_hole, &ndisy_hole, &ndisz_hole);				   
						   
	float posx_site = d_posx[pid] + ndisx_hole * std_hole; 
	float posy_site = d_posy[pid] + ndisy_hole * std_hole; 
	float posz_site = d_posz[pid] + ndisz_hole * std_hole; 
		   
	std_dis = d_placeinfo_btype[btype*9 + 1]/__fsqrt_rn(3.0f);
	getNormalizedDis_Sample3DGuassian(localState_pt, &ndisx, &ndisy, &ndisz);	
		   
	float coef = d_placeinfo_btype[btype*9 + 3];
	
	//float posx_dis = posx_site + coef * ndisx * std_dis;
	//float posy_dis = posy_site + coef * ndisy * std_dis;
	//float posz_dis = posz_site + coef * ndisz * std_dis;
		   
	float radnum = curand_uniform(localState_pt);
	
	//printf("displace_twoproducts_oneelec_holehoping: pid = %d, radnum = %f, ndisx_hole = %f, ndisy_hole = %f, ndisz_hole = %f, std_hole = %f, ndisx = %f, ndisy = %f, ndisz = %f, std_dis = %f\n", pid, radnum, ndisx_hole, ndisy_hole, ndisz_hole, std_hole, ndisx, ndisy, ndisz, std_dis);
	
	if(radnum < 0.5f) // put the two products randomly
	{
	  d_posx[pid] = posx_site;
	  d_posy[pid] = posy_site;
	  d_posz[pid] = posz_site;
	  
	  d_posx[pid + d_num_total] = posx_site + coef * ndisx * std_dis;
	  d_posy[pid + d_num_total] = posy_site + coef * ndisy * std_dis;
	  d_posz[pid + d_num_total] = posz_site + coef * ndisz * std_dis;
	}
	else
	{
	  d_posx[pid + d_num_total] = posx_site;
	  d_posy[pid + d_num_total] = posy_site;
	  d_posz[pid + d_num_total] = posz_site;
	  
	  d_posx[pid] = posx_site + coef * ndisx * std_dis;
	  d_posy[pid] = posy_site + coef * ndisy * std_dis;
	  d_posz[pid] = posz_site + coef * ndisz * std_dis;
	}

    //place the third production that is an electron, with its thermal energy considered to be 0.025 eV to obtain its thermal distance
    float idx_ebin = d_ide_ebin*( 0.025f - d_mine_ebin) + 0.5f;
	sampleThermalDistance(pid, localState_pt, &ndisx, &ndisy, &ndisz, idx_ebin);	
	
	d_posx[pid + d_num_total * 2] = posx_site + ndisx;
	d_posy[pid + d_num_total * 2] = posy_site + ndisy;
	d_posz[pid + d_num_total * 2] = posz_site + ndisz;
	
	//printf("displace_twoproducts_oneelec_holehoping: pid = %d, ndisx = %f, ndisy = %f, ndisz = %f\n", pid, ndisx, ndisy, ndisz);
}														

__device__ void getNormalizedDis_Sample3DGuassian(curandState *localState_pt, float *ndisx, float *ndisy, float *ndisz)                                                               					
{// sampling accoding to a 3D Gaussian
	float zeta1 = -2.0f * log(curand_uniform(localState_pt));
    float zeta2 = 2.0f * PI * curand_uniform(localState_pt);
	
	*ndisx = __fsqrt_rn(zeta1)* __cosf(zeta2);
	*ndisy = __fsqrt_rn(zeta1)* __sinf(zeta2);
		
	zeta1 = -2.0f * log(curand_uniform(localState_pt));
	zeta2 = 2.0f * PI * curand_uniform(localState_pt);
		
	*ndisz = __fsqrt_rn(zeta1)* __cosf(zeta2);    
}

__device__ void getDirection_SampleOnSphereSurface(curandState *localState_pt, float *ndisx, float *ndisy, float *ndisz)                                                               					
{// uniform sampling on a unit sphere
	float beta = curand_uniform(localState_pt)*2.0f*PI;
	float costheta = 1.0f-2.0f*curand_uniform(localState_pt);
	
	*ndisx = sqrtf(1-costheta*costheta) * __cosf(beta);
	*ndisy = sqrtf(1-costheta*costheta)  * __sinf(beta);
	*ndisz = costheta;
}												  

void PrechemList::initGPUVariables()
{
	float *temp_pos = (float*)malloc(sizeof(float) * num_total * 2);
	for(int i=0; i<num_total * 2; i++)
		temp_pos[i] = 10000;
	cudaMalloc((void **) &d_posx, sizeof(float)* num_total * 3); // one initial particle may produce up to 3 products, so triple memory size to include the products
	cudaMemcpy(d_posx, posx, sizeof(float)*num_total, cudaMemcpyHostToDevice);
	cudaMemcpy(d_posx+num_total, temp_pos, sizeof(float)*num_total*2, cudaMemcpyHostToDevice);
	
	cudaMalloc((void **) &d_posy, sizeof(float)* num_total * 3);
	cudaMemcpy(d_posy, posy, sizeof(float)*num_total, cudaMemcpyHostToDevice);
	cudaMemcpy(d_posy+num_total, temp_pos, sizeof(float)*num_total*2, cudaMemcpyHostToDevice);
	
	cudaMalloc((void **) &d_posz, sizeof(float)* num_total * 3);
	cudaMemcpy(d_posz, posz, sizeof(float)*num_total, cudaMemcpyHostToDevice);
	cudaMemcpy(d_posz+num_total, temp_pos, sizeof(float)*num_total*2, cudaMemcpyHostToDevice);
	free(temp_pos);
	
	cudaMalloc((void **) &d_index, sizeof(int)* num_total * 3);
	cudaMemcpy(d_index, index, sizeof(int)*num_total, cudaMemcpyHostToDevice);
	cudaMemcpy(d_index+num_total, index, sizeof(int)*num_total, cudaMemcpyHostToDevice);
	cudaMemcpy(d_index+num_total*2, index, sizeof(int)*num_total, cudaMemcpyHostToDevice);
	
	cudaMalloc((void **) &d_ttime, sizeof(float)* num_total * 3);
	cudaMemcpy(d_ttime, ttime, sizeof(float)*num_total, cudaMemcpyHostToDevice);
	cudaMemcpy(d_ttime+num_total, ttime, sizeof(float)*num_total, cudaMemcpyHostToDevice);
	cudaMemcpy(d_ttime+num_total*2, ttime, sizeof(float)*num_total, cudaMemcpyHostToDevice);

	int *temp_ptype = (int*)malloc(sizeof(int) * num_total * 3);
	for(int i=0; i<num_total * 3; i++)
		temp_ptype[i] = 255;	
	cudaMalloc((void **) &d_ptype, sizeof(int)* num_total * 3);
	cudaMemcpy(d_ptype, temp_ptype, sizeof(float)*num_total*3, cudaMemcpyHostToDevice);
	free(temp_ptype);
	
	cudaMalloc((void **) &d_statetype, sizeof(int)* num_total);
	cudaMemcpy(d_statetype, statetype, sizeof(int)*num_total, cudaMemcpyHostToDevice);
	
	cudaMalloc((void **) &d_ene, sizeof(float)* num_total); //since energy will not be considered at chemical stage, so only for the initial particles of the prechemical stage
	cudaMemcpy(d_ene, ene, sizeof(float)*num_total, cudaMemcpyHostToDevice);
	
	cudaMalloc((void **) &d_wiid_elec, sizeof(int)* num_elec); 
	cudaMemcpy(d_wiid_elec, wiid_elec, sizeof(int)*num_elec, cudaMemcpyHostToDevice);
	
    cudaMemcpyToSymbol(d_num_elec, &num_elec, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_num_wi, &num_wi, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_num_we_a1b1, &num_we_a1b1, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_num_we_b1a1, &num_we_b1a1, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_num_we_rd, &num_we_rd, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_num_w_dis, &num_w_dis, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_num_total, &num_total, sizeof(int), 0, cudaMemcpyHostToDevice);
	
	cudaMemcpyToSymbol(d_sidx_elec, &sidx_elec, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_sidx_wi, &sidx_wi, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_sidx_we_a1b1, &sidx_we_a1b1, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_sidx_we_b1a1, &sidx_we_b1a1, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_sidx_we_rd, &sidx_we_rd, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_sidx_w_dis, &sidx_w_dis, sizeof(int), 0, cudaMemcpyHostToDevice);

	cudaMemcpyToSymbol(d_nebin, &nebin, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_mine_ebin, &mine_ebin, sizeof(float), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_ide_ebin, &ide_ebin, sizeof(float), 0, cudaMemcpyHostToDevice);
	
	cudaMemcpyToSymbol(d_nbtype, &nbtype, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_placeinfo_btype, placeinfo_btype, sizeof(float)*nbtype*9, 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_num_product_btype, num_product_btype, sizeof(int)*nbtype, 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_ptype_product_btype, ptype_product_btype, sizeof(int)*nbtype * MAXNUMBRANCHPROD, 0, cudaMemcpyHostToDevice);
	
	cudaMemcpyToSymbol(d_nb_rece, &nb_rece, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_nb_wi, &nb_wi, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_nb_we_a1b1, &nb_we_a1b1, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_nb_we_b1a1, &nb_we_b1a1, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_nb_we_rd, &nb_we_rd, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_nb_w_dis, &nb_w_dis, sizeof(int), 0, cudaMemcpyHostToDevice);
	
	cudaMemcpyToSymbol(d_pb_rece, pb_rece, sizeof(float)*nb_rece, 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_pb_wi, pb_wi, sizeof(float)*nb_wi, 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_pb_we_a1b1, pb_we_a1b1, sizeof(float)*nb_we_a1b1, 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_pb_we_b1a1, pb_we_b1a1, sizeof(float)*nb_we_b1a1, 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_pb_we_rd, pb_we_rd, sizeof(float)*nb_we_rd, 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_pb_w_dis, pb_w_dis, sizeof(float)*nb_w_dis, 0, cudaMemcpyHostToDevice);
	
    cudaMemcpyToSymbol(d_btype_rece, btype_rece, sizeof(int)*nb_rece, 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_btype_wi, btype_wi, sizeof(int)*nb_wi, 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_btype_we_a1b1, btype_we_a1b1, sizeof(int)*nb_we_a1b1, 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_btype_we_b1a1, btype_we_b1a1, sizeof(int)*nb_we_b1a1, 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_btype_we_rd, btype_we_rd, sizeof(int)*nb_we_rd, 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_btype_w_dis, btype_w_dis, sizeof(int)*nb_w_dis, 0, cudaMemcpyHostToDevice);
 
    cudaMallocArray(&d_p_recomb_elec, &p_recomb_elec_tex.channelDesc, nebin, 1);
    cudaMemcpyToArray(d_p_recomb_elec, 0, 0, p_recomb_elec, sizeof(float)*nebin, cudaMemcpyHostToDevice);
    p_recomb_elec_tex.filterMode = cudaFilterModeLinear;
    cudaBindTextureToArray(p_recomb_elec_tex, d_p_recomb_elec);
	
	cudaMallocArray(&d_rms_therm_elec, &rms_therm_elec_tex.channelDesc, nebin, 1);
    cudaMemcpyToArray(d_rms_therm_elec, 0, 0, rms_therm_elec, sizeof(float)*nebin, cudaMemcpyHostToDevice);
    rms_therm_elec_tex.filterMode = cudaFilterModeLinear;
    cudaBindTextureToArray(rms_therm_elec_tex, d_rms_therm_elec);

	for(int i=0; i<5; i++)
	    cudaStreamCreate(&stream[i]);
}

void PrechemList::run()
{
	//simulating the prechemical stage for the subexcitation electrons: thermalisation or recombination with its parent ionized water
    int nblocks = 1 + (num_elec - 1)/NTHREAD_PER_BLOCK_PAR;
	thermalisation_subexelectrons<<<nblocks,NTHREAD_PER_BLOCK_PAR, 0, stream[0]>>>(d_posx, d_posy, d_posz, d_ene, d_ptype, d_statetype, d_wiid_elec);
	cudaStreamSynchronize(stream[0]);
	
	//simulating the prechemical stage for the ionized water (the ones don't have recombination with water
	nblocks = 1 + (num_wi - 1)/NTHREAD_PER_BLOCK_PAR;
	dissociation_ionizedwater<<<nblocks, NTHREAD_PER_BLOCK_PAR, 0, stream[0]>>>(d_posx, d_posy, d_posz, d_ptype, d_statetype); 
	
	//simulating the prechemical stage for the excited water with A1B1 excitation state
	nblocks = 1 + (num_we_a1b1 - 1)/NTHREAD_PER_BLOCK_PAR;
	dissociation_excitedwater_a1b1<<<nblocks, NTHREAD_PER_BLOCK_PAR, 0, stream[1]>>>(d_posx, d_posy, d_posz, d_ptype, d_statetype); 
	
	//simulating the prechemical stage for the excited water with A1B1 excitation state
	nblocks = 1 + (num_we_b1a1 - 1)/NTHREAD_PER_BLOCK_PAR;
	dissociation_excitedwater_b1a1<<<nblocks, NTHREAD_PER_BLOCK_PAR, 0, stream[2]>>>(d_posx, d_posy, d_posz, d_ptype, d_statetype); 
	
	//simulating the prechemical stage for the excited water with A1B1 excitation state
	nblocks = 1 + (num_we_rd - 1)/NTHREAD_PER_BLOCK_PAR;
	dissociation_excitedwater_rd<<<nblocks, NTHREAD_PER_BLOCK_PAR, 0, stream[3]>>>(d_posx, d_posy, d_posz, d_ptype, d_statetype); 
	
	//simulating the prechemical stage for the excited water with A1B1 excitation state
	nblocks = 1 + (num_w_dis - 1)/NTHREAD_PER_BLOCK_PAR;
	dissociation_dissociativewater<<<nblocks, NTHREAD_PER_BLOCK_PAR, 0, stream[4]>>>(d_posx, d_posy, d_posz, d_ptype, d_statetype);
	
	cudaDeviceSynchronize();
		
}

void PrechemList::saveResults()
{
	FILE *fp;
	
	//remove the empty entries or H2O entries from the particle data
	thrust::device_ptr<float> posx_dev_ptr;
	thrust::device_ptr<float> posy_dev_ptr;
	thrust::device_ptr<float> posz_dev_ptr;
	thrust::device_ptr<int> ptype_dev_ptr;
	thrust::device_ptr<int> index_dev_ptr;
	thrust::device_ptr<float> ttime_dev_ptr;
	
	typedef thrust::tuple<thrust::device_vector<int>::iterator, thrust::device_vector<float>::iterator, thrust::device_vector<float>::iterator, thrust::device_vector<float>::iterator, thrust::device_vector<int>::iterator, thrust::device_vector<float>::iterator> IteratorTuple;
        // define a zip iterator
	typedef thrust::zip_iterator<IteratorTuple> ZipIterator;
	
	ZipIterator zip_begin, zip_end, zip_new_end;
	
	ptype_dev_ptr = thrust::device_pointer_cast(&d_ptype[0]);		
	posx_dev_ptr = thrust::device_pointer_cast(&d_posx[0]);	
	posy_dev_ptr = thrust::device_pointer_cast(&d_posy[0]);	
	posz_dev_ptr = thrust::device_pointer_cast(&d_posz[0]);	
	index_dev_ptr = thrust::device_pointer_cast(&d_index[0]);
	ttime_dev_ptr = thrust::device_pointer_cast(&d_ttime[0]);

	zip_begin = thrust::make_zip_iterator(thrust::make_tuple(ptype_dev_ptr, posx_dev_ptr, posy_dev_ptr, posz_dev_ptr, index_dev_ptr, ttime_dev_ptr));
	zip_end   = zip_begin + num_total * 3;  		
	zip_new_end = thrust::remove_if(zip_begin, zip_end, first_element_equal_255());
	
	cudaDeviceSynchronize();
	
	int	numCurPar = zip_new_end - zip_begin;
		
	printf("After removing, numCurPar = %d\n", numCurPar);
	float *output_posx = (float*)malloc(sizeof(float) * numCurPar);
    float *output_posy = (float*)malloc(sizeof(float) * numCurPar);
    float *output_posz = (float*)malloc(sizeof(float) * numCurPar);
    float *output_ttime = (float*)malloc(sizeof(float) * numCurPar);
    int *output_ptype = (int*)malloc(sizeof(float) * numCurPar);
    int *output_index = (int*)malloc(sizeof(float) * numCurPar);
    
    cudaMemcpyAsync(output_posx , d_posx, sizeof(float)*numCurPar, cudaMemcpyDeviceToHost, stream[0]);	
    cudaMemcpyAsync(output_posy , d_posy, sizeof(float)*numCurPar, cudaMemcpyDeviceToHost, stream[1]);	
    cudaMemcpyAsync(output_posz , d_posz, sizeof(float)*numCurPar, cudaMemcpyDeviceToHost, stream[2]);
    cudaMemcpyAsync(output_ptype, d_ptype, sizeof(int)*numCurPar, cudaMemcpyDeviceToHost, stream[3]);	
    cudaMemcpyAsync(output_index, d_index, sizeof(int)*numCurPar, cudaMemcpyDeviceToHost, stream[4]);	
    cudaMemcpyAsync(output_ttime, d_ttime, sizeof(int)*numCurPar, cudaMemcpyDeviceToHost, stream[4]);	

	cudaDeviceSynchronize();
	
	std::string fname = document["fileForOutput"].GetString();
	fp = fopen(fname.c_str(), "wb");	
    fwrite(output_posx, sizeof(float), numCurPar, fp);
    fwrite(output_posy, sizeof(float), numCurPar, fp);
	fwrite(output_posz, sizeof(float), numCurPar, fp);
	fwrite(output_ttime, sizeof(float), numCurPar, fp);
	fwrite(output_index, sizeof(int), numCurPar, fp);
	fwrite(output_ptype, sizeof(int), numCurPar, fp);
	fclose(fp);	
	

    cudaFree(d_posx);
    cudaFree(d_posy);	 
    cudaFree(d_posy);	
	cudaFree(d_ptype);
    cudaFree(d_statetype);	
    cudaFree(d_wiid_elec);
	cudaFree(d_ene);
    cudaFree(d_ttime);
    cudaFree(d_index);

    cudaUnbindTexture(p_recomb_elec_tex);
    cudaFreeArray(d_p_recomb_elec);
    cudaUnbindTexture(rms_therm_elec_tex);
    cudaFreeArray(d_rms_therm_elec);

	for(int i=0; i<5; i++)
	    cudaStreamDestroy(stream[i]);
}
