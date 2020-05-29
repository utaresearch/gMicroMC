#ifndef __MICROMC_PC_KERNELS_H__
#define __MICROMC_PC_KERNELS_H__

#include <stdio.h>

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
	     //float idx_ebin = 0.5f; //use 24.8% constant recombination rate for the electrons		 
		 
		 float temp = tex1D(p_recomb_elec_tex, idx_ebin);
	     //float temp = 0.26f;
		 
         //printf("thermalisation_subexelectrons: pid = %d, temp = %f, radnum = %f\n", pid, temp, radnum);	
		 
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
	// only one branch for ionized water 
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
	
	//float posx_dis = posx_site + coef * ndisx * std_dis;
	//float posy_dis = posy_site + coef * ndisy * std_dis;
	//float posz_dis = posz_site + coef * ndisz * std_dis;
		   
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
	float radnum = curand_uniform(localState_pt) * MAXPVALUE;		  
	float temp = 4.0f*r*__expf(-2.0*r);
	  
	while(radnum>temp)
	{
	    r  = curand_uniform(localState_pt) * 4.5f; 
		radnum = curand_uniform(localState_pt) * MAXPVALUE;
		temp = 4.0f*r*__expf(-2.0*r);
		
		//printf("rejection: tid = %d, radnum = %f, temp = %f, r = %f\n", tid, radnum, temp, r);
	}
	  
	float rms_therm = tex1D(rms_therm_elec_tex, idx_ebin); //scaled by thermalization rms displacement	
	  
	//printf("final: tid = %d, ene = %f, idx_ebin = %f, r = %f\n", tid, d_ene[pid], idx_ebin, r);

	getDirection_SampleOnSphereSurface(localState_pt, ndisx, ndisy, ndisz); 
	
	//printf("sampleThermalDistance: pid = %d, idx_ebin = %f, radnum = %f, temp = %f, r = %f, rms_therm = %f, ndisx = %f, ndisy = %f, ndisz = %f\n", pid, idx_ebin, radnum, temp, r, rms_therm, *ndisx, *ndisy, *ndisz);
	
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

    //place the third production that is an electron, with its thermal energy considered to be 0.025 eV to obtain its thermal distance
    float idx_ebin = d_ide_ebin*( 0.025f - d_mine_ebin) + 0.5f;
	sampleThermalDistance(pid, localState_pt, &ndisx, &ndisy, &ndisz, idx_ebin);	
	
	d_posx[pid + d_num_total * 2] = posx_site + ndisx;
	d_posy[pid + d_num_total * 2] = posy_site + ndisy;
	d_posz[pid + d_num_total * 2] = posz_site + ndisz;
	
	//printf("displace_twoproducts_oneelec_holehoping: pid = %d, ndisx = %f, ndisy = %f, ndisz = %f\n", pid, ndisx, ndisy, ndisz);
}														

__device__ void getNormalizedDis_Sample3DGuassian(curandState *localState_pt, float *ndisx, float *ndisy, float *ndisz)                                                               					
{
	float zeta1 = -2.0f * log(curand_uniform(localState_pt));
    float zeta2 = 2.0f * PI * curand_uniform(localState_pt);
	
	*ndisx = __fsqrt_rn(zeta1)* __cosf(zeta2);
	*ndisy = __fsqrt_rn(zeta1)* __sinf(zeta2);
		
	zeta1 = -2.0f * log(curand_uniform(localState_pt));
	zeta2 = 2.0f * PI * curand_uniform(localState_pt);
		
	*ndisz = __fsqrt_rn(zeta1)* __cosf(zeta2);    
}

__device__ void getDirection_SampleOnSphereSurface(curandState *localState_pt, float *ndisx, float *ndisy, float *ndisz)                                                               					
{
	float beta = curand_uniform(localState_pt)*2.0f*PI;
	float costheta = 1.0f-2.0f*curand_uniform(localState_pt);
	
	*ndisx = sqrtf(1-costheta*costheta) * __cosf(beta);
	*ndisy = sqrtf(1-costheta*costheta)  * __sinf(beta);
	*ndisz = costheta;
}												  
#endif
