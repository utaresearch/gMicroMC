#include <stdlib.h>
#include <string>
#include <iostream>
#include <sstream>
using namespace std;
#include <ctime>
#include <cuda.h>
#include <cuda_runtime.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/binary_search.h>
#include <thrust/functional.h>
#include <thrust/count.h>
#include <thrust/extrema.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/gather.h>
#include <thrust/unique.h>
#include <thrust/uninitialized_copy.h>
#include "realtime.cu"

#include <cusparse.h>

#include "microMC_chem.h"

struct first_element_equal_255
{
  __host__ __device__
  bool operator()(const thrust::tuple<const unsigned char&, const float&, const float&, const float&, const float&, const int&> &t)
  {
      return thrust::get<0>(t) == 255;
  }
};

void runMicroMC(ChemistrySpec *chemistrySpec, ReactionType *reactType, ParticleData *parData, int process_time, int flagDNA)
{	
    float max_posx, min_posx, max_posy, min_posy, max_posz, min_posz, mintd;
	
	float binSize, binSize_diffu;
    unsigned long numBinx, numBiny, numBinz,  numNZBin;//numBin,
	
	thrust::device_ptr<float> max_ptr;
	thrust::device_ptr<float> min_ptr;
		
	cusparseStatus_t status;
	cusparseHandle_t handle=0;
	cusparseMatDescr_t descra=0;
	status= cusparseCreate(&handle);
	status= cusparseCreateMatDescr(&descra);
	cusparseSetMatType(descra,CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descra,CUSPARSE_INDEX_BASE_ZERO);
	
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

	thrust::device_vector<unsigned long> uniBinidxPar_dev_vec(MAXNUMPAR);
			
	typedef thrust::tuple<thrust::device_vector<unsigned char>::iterator, thrust::device_vector<float>::iterator, thrust::device_vector<float>::iterator, thrust::device_vector<float>::iterator,thrust::device_vector<float>::iterator,thrust::device_vector<int>::iterator> IteratorTuple;
        // define a zip iterator
		
	typedef thrust::zip_iterator<IteratorTuple> ZipIterator;
		
	ZipIterator zip_begin, zip_end, zip_new_end;
	
	int idx_iter = 0;	
	
	int nblocks;
	int idx_typedeltaT;
	
	int idx_neig = 0;
	float numofextendbin=5;
	
	float* h_posx=NULL;
    float* h_posy=NULL;
    float* h_posz=NULL;
    float* h_ttime=NULL;
    int* h_index = NULL;
    unsigned char* h_ptype=NULL;
    h_posx=(float*) malloc(sizeof(float) * numCurPar*2);
	h_posy=(float*) malloc(sizeof(float) * numCurPar*2);
	h_posz=(float*) malloc(sizeof(float) * numCurPar*2);
	h_ttime = (float*) malloc(sizeof(float) * numCurPar*2);
	h_index = (int*) malloc(sizeof(int) * numCurPar*2);
	h_ptype=(unsigned char*) malloc(sizeof(unsigned char) * numCurPar*2);

/***********************************************************************************/	
	while(curTime < process_time) //curTime starts from 1
	{
		if(numCurPar==0) break;
		//printf("------------------------------ : \n");
		//printf("Begin the simulation of the %dth time step: \n", idx_iter);
		//printf("------------------------------ : \n");		
		if(curTime < 10.0f)
		    idx_typedeltaT = 0;
		else if(curTime < 100.0f)
		    idx_typedeltaT = 1;
		else if(curTime < 1000.0f)
		   idx_typedeltaT = 2;
		else if(curTime < 10000.0f)
		   idx_typedeltaT = 3;
		else
			idx_typedeltaT = 4;
					
		h_deltaT = reactType->h_deltaT_adap[idx_typedeltaT];
		
		binSize = 2 * reactType->max_calc_radii_React[idx_typedeltaT];
        if(idx_iter == 0)
        {		
			posx_dev_ptr = thrust::device_pointer_cast(&d_posx[0]);
			max_ptr = thrust::max_element(posx_dev_ptr, posx_dev_ptr + numCurPar);
			max_posx=max_ptr[0]+numofextendbin;
			min_ptr = thrust::min_element(posx_dev_ptr, posx_dev_ptr + numCurPar);
			min_posx=min_ptr[0]-numofextendbin;
			
			posy_dev_ptr = thrust::device_pointer_cast(&d_posy[0]);
			max_ptr = thrust::max_element(posy_dev_ptr, posy_dev_ptr + numCurPar);
			max_posy=max_ptr[0]+numofextendbin;
			min_ptr = thrust::min_element(posy_dev_ptr, posy_dev_ptr + numCurPar);
			min_posy=min_ptr[0]-numofextendbin;
				
			posz_dev_ptr = thrust::device_pointer_cast(&d_posz[0]);
			max_ptr = thrust::max_element(posz_dev_ptr, posz_dev_ptr + numCurPar);
			max_posz=max_ptr[0]+numofextendbin;
			min_ptr = thrust::min_element(posz_dev_ptr, posz_dev_ptr + numCurPar);
			min_posz=min_ptr[0]-numofextendbin;
			
			printf("max_posx = %f, min_posx = %f, max_posy = %f, min_posy = %f, max_posz = %f, min_posz = %f\n", max_posx, min_posx, max_posy, min_posy, max_posz, min_posz);			
		}

		numBinx = (max_posx - min_posx)/binSize + 1;
		numBiny = (max_posy - min_posy)/binSize + 1;
		numBinz = (max_posz - min_posz)/binSize + 1;
		
		nblocks = 1 + (numCurPar - 1)/NTHREAD_PER_BLOCK_PAR;
		assignBinidx4Par<<<nblocks,NTHREAD_PER_BLOCK_PAR>>>(d_gridParticleHash, d_gridParticleIndex, d_posx, d_posy, d_posz, min_posx, min_posy, min_posz, numBinx, numBiny, numBinz, binSize, numCurPar);		
		cudaDeviceSynchronize();		
				
		gridHash_dev_ptr = thrust::device_pointer_cast(&d_gridParticleHash[0]);
		gridIndex_dev_ptr = thrust::device_pointer_cast(&d_gridParticleIndex[0]);
		thrust::sort_by_key(gridHash_dev_ptr, gridHash_dev_ptr + numCurPar, gridIndex_dev_ptr);
		
		
		result_unique_copy = thrust::unique_copy(gridHash_dev_ptr, gridHash_dev_ptr + numCurPar, uniBinidxPar_dev_vec.begin());
		
		numNZBin = result_unique_copy - uniBinidxPar_dev_vec.begin();
		//printf("numNZBin = %d\n", numNZBin);
		
		d_nzBinidx =  thrust::raw_pointer_cast(&uniBinidxPar_dev_vec[0]); 	

		nblocks = 1 + (numCurPar - 1)/NTHREAD_PER_BLOCK_PAR;
		FindParIdx4NonZeroBin<<<nblocks, NTHREAD_PER_BLOCK_PAR>>>(d_gridParticleHash, d_nzBinidx, d_accumParidxBin, numNZBin,numCurPar);
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
		
		nblocks = 1 + (numNZBin * 27 - 1)/NTHREAD_PER_BLOCK_PAR;
		FindNeig4NonZeroBin<<<nblocks, NTHREAD_PER_BLOCK_PAR>>>(d_nzBinidx, d_idxnzBin_neig, d_idxnzBin_numNeig, numNZBin);
		cudaDeviceSynchronize();
		//printf("FindNeig4NonZeroBin kernel is done\n");	
	
		CUDA_CALL(cudaBindTexture(0, posx_tex, d_posx, sizeof(float) * numCurPar));
		CUDA_CALL(cudaBindTexture(0, posy_tex, d_posy, sizeof(float) * numCurPar));
		CUDA_CALL(cudaBindTexture(0, posz_tex, d_posz, sizeof(float) * numCurPar));
		CUDA_CALL(cudaBindTexture(0, ptype_tex, d_ptype, sizeof(unsigned char) * numCurPar));
		
		nblocks = 1 + (numCurPar - 1)/NTHREAD_PER_BLOCK_PAR;
		reorderData_beforeDiffusion<<<nblocks,NTHREAD_PER_BLOCK_PAR>>>(d_posx_s, d_posy_s, d_posz_s, d_ptype_s,d_gridParticleIndex, numCurPar);                                        
		//assign id after sorted
		cudaDeviceSynchronize();

		CUDA_CALL(cudaUnbindTexture(posx_tex));
		CUDA_CALL(cudaUnbindTexture(posy_tex));
		CUDA_CALL(cudaUnbindTexture(posz_tex));
		CUDA_CALL(cudaUnbindTexture(ptype_tex));
		
		//printf("reorderData before Diffusion is done\n");	     
		
		CUDA_CALL(cudaMemset(d_statusPar, 255, sizeof(unsigned char) * iniPar));
		CUDA_CALL(cudaMemset(d_statusPar, 0, sizeof(unsigned char) * numCurPar));
		CUDA_CALL(cudaMemset(d_ptype, 255, sizeof(unsigned char) * iniPar)); // use 255 to mark the void entry in the new particle array

		CUDA_CALL(cudaMemcpy(d_ptype, d_ptype_s, sizeof(unsigned char) * numCurPar, cudaMemcpyDeviceToDevice));
		CUDA_CALL(cudaMemcpy(d_posx, d_posx_s, sizeof(float) * numCurPar, cudaMemcpyDeviceToDevice));
		CUDA_CALL(cudaMemcpy(d_posy, d_posy_s, sizeof(float) * numCurPar, cudaMemcpyDeviceToDevice));
		CUDA_CALL(cudaMemcpy(d_posz, d_posz_s, sizeof(float) * numCurPar, cudaMemcpyDeviceToDevice));
	
		CUDA_CALL(cudaMemcpy(d_mintd_Par, h_mintd_Par_init, sizeof(float)*numCurPar, cudaMemcpyHostToDevice));//min time, initilized to 1e6
		
		CUDA_CALL(cudaBindTexture(0, posx_tex, d_posx_s, sizeof(float) * numCurPar));
		CUDA_CALL(cudaBindTexture(0, posy_tex, d_posy_s, sizeof(float) * numCurPar));
		CUDA_CALL(cudaBindTexture(0, posz_tex, d_posz_s, sizeof(float) * numCurPar));
		CUDA_CALL(cudaBindTexture(0, ptype_tex, d_ptype_s, sizeof(unsigned char) * numCurPar));

		nblocks = 1 + (numCurPar - 1)/NTHREAD_PER_BLOCK_PAR;
		react4TimeStep_beforeDiffusion<<<nblocks, NTHREAD_PER_BLOCK_PAR>>>(d_posx, d_posy, d_posz, d_ptype, d_gridParticleHash, d_idxnzBin_neig, d_idxnzBin_numNeig, d_nzBinidx, d_accumParidxBin, d_statusPar, d_mintd_Par, numBinx, numBiny, numBinz, numNZBin, numCurPar, idx_typedeltaT);
		cudaDeviceSynchronize();

		CUDA_CALL(cudaUnbindTexture(posx_tex));
		CUDA_CALL(cudaUnbindTexture(posy_tex));
		CUDA_CALL(cudaUnbindTexture(posz_tex));
		CUDA_CALL(cudaUnbindTexture(ptype_tex));			
		
		mintd_dev_ptr = thrust::device_pointer_cast(d_mintd_Par);
		min_ptr = thrust::min_element(mintd_dev_ptr, mintd_dev_ptr + numCurPar);
		mintd = min_ptr[0];		
		//printf("mintd = %f\n", mintd);

	    if(mintd > 0.0f) // some reactions occurs before diffusion, so no diffusion at this time step, delta t = 0
	    {
			if(mintd < h_deltaT || mintd >= 10000.0f)
			   mintd = h_deltaT;
			
			curTime += mintd;
			//printf("curTime = %f  mintd = %f\n", curTime, mintd);
			   
			cudaMemcpyToSymbol(d_deltaT, &mintd, sizeof(float), 0, cudaMemcpyHostToDevice);
			
			CUDA_CALL(cudaBindTexture(0, posx_tex, d_posx, sizeof(float) * numCurPar));//update d_posx in the above codes
			CUDA_CALL(cudaBindTexture(0, posy_tex, d_posy, sizeof(float) * numCurPar));
			CUDA_CALL(cudaBindTexture(0, posz_tex, d_posz, sizeof(float) * numCurPar));
			CUDA_CALL(cudaBindTexture(0, ptype_tex, d_ptype, sizeof(unsigned char) * numCurPar));
			
			nblocks = 1 + (numCurPar - 1)/NTHREAD_PER_BLOCK_PAR;
			makeOneJump4Diffusion<<<nblocks, NTHREAD_PER_BLOCK_PAR>>>(d_posx_d, d_posy_d, d_posz_d,numCurPar);
			cudaDeviceSynchronize();
		
			binSize_diffu = sqrt(6.0f * chemistrySpec->maxDiffCoef_spec * mintd); 
			
			if(binSize < binSize_diffu)
			{
			    binSize = binSize_diffu;	    
			}
			/*******************revised at Mar 3rd 2019. After diffusion, the minimun and maximum position should be updated***
			if not updated, the linear index can be wrong! (binidx_x can be larger than numBinx)*/
			posx_dev_ptr = thrust::device_pointer_cast(&d_posx_d[0]);
			max_ptr = thrust::max_element(posx_dev_ptr, posx_dev_ptr + numCurPar);
			max_posx=max_ptr[0]+numofextendbin;
			min_ptr = thrust::min_element(posx_dev_ptr, posx_dev_ptr + numCurPar);
			min_posx=min_ptr[0]-numofextendbin;
			
			posy_dev_ptr = thrust::device_pointer_cast(&d_posy_d[0]);
			max_ptr = thrust::max_element(posy_dev_ptr, posy_dev_ptr + numCurPar);
			max_posy=max_ptr[0]+numofextendbin;
			min_ptr = thrust::min_element(posy_dev_ptr, posy_dev_ptr + numCurPar);
			min_posy=min_ptr[0]-numofextendbin;
				
			posz_dev_ptr = thrust::device_pointer_cast(&d_posz_d[0]);
			max_ptr = thrust::max_element(posz_dev_ptr, posz_dev_ptr + numCurPar);
			max_posz=max_ptr[0]+numofextendbin;
			min_ptr = thrust::min_element(posz_dev_ptr, posz_dev_ptr + numCurPar);
			min_posz=min_ptr[0]-numofextendbin;			
			//printf("max_posx = %f, min_posx = %f, max_posy = %f, min_posy = %f, max_posz = %f, min_posz = %f\n", max_posx, min_posx, max_posy, min_posy, max_posz, min_posz);			
			
	        numBinx = (max_posx - min_posx)/binSize + 1;
			numBiny = (max_posy - min_posy)/binSize + 1;
			numBinz = (max_posz - min_posz)/binSize + 1;
			//numBin = numBinx * numBiny * numBinz;
		
			//printf("numBinx = %lu, numBiny = %lu, numBinz = %lu, numBin = %lu, binSize = %f\n", numBinx, numBiny, numBinz, numBin, binSize);
				
			nblocks = 1 + (numCurPar - 1)/NTHREAD_PER_BLOCK_PAR;
			assignBinidx4Par<<<nblocks,NTHREAD_PER_BLOCK_PAR>>>(d_gridParticleHash, d_gridParticleIndex, d_posx_d, d_posy_d, d_posz_d, min_posx, min_posy, min_posz, numBinx, numBiny, numBinz, binSize, numCurPar);			
			cudaDeviceSynchronize();	
		
			gridHash_dev_ptr = thrust::device_pointer_cast(&d_gridParticleHash[0]);
			gridIndex_dev_ptr = thrust::device_pointer_cast(&d_gridParticleIndex[0]);
			thrust::sort_by_key(gridHash_dev_ptr, gridHash_dev_ptr + numCurPar, gridIndex_dev_ptr);
		
			result_unique_copy = thrust::unique_copy(gridHash_dev_ptr, gridHash_dev_ptr + numCurPar, uniBinidxPar_dev_vec.begin());
			
			numNZBin = result_unique_copy - uniBinidxPar_dev_vec.begin();
			//printf("numNZBin = %d\n", numNZBin);
			
			d_nzBinidx =  thrust::raw_pointer_cast(&uniBinidxPar_dev_vec[0]); 	
				
		    nblocks = 1 + (numCurPar - 1)/NTHREAD_PER_BLOCK_PAR;
			FindParIdx4NonZeroBin<<<nblocks, NTHREAD_PER_BLOCK_PAR>>>(d_gridParticleHash, d_nzBinidx, d_accumParidxBin, numNZBin,numCurPar);
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
			
			nblocks = 1 + (numNZBin * 27 - 1)/NTHREAD_PER_BLOCK_PAR;
			FindNeig4NonZeroBin<<<nblocks, NTHREAD_PER_BLOCK_PAR>>>(d_nzBinidx, d_idxnzBin_neig, d_idxnzBin_numNeig, numNZBin);
			cudaDeviceSynchronize();

			cudaBindTexture(0, posx_d_tex, d_posx_d, sizeof(float) * numCurPar);
			cudaBindTexture(0, posy_d_tex, d_posy_d, sizeof(float) * numCurPar);
			cudaBindTexture(0, posz_d_tex, d_posz_d, sizeof(float) * numCurPar);
			
			nblocks = 1 + (numCurPar - 1)/NTHREAD_PER_BLOCK_PAR;
			reorderData_afterDiffusion<<<nblocks,NTHREAD_PER_BLOCK_PAR>>>(d_posx_s, d_posy_s, d_posz_s, d_ptype_s, d_posx_sd, d_posy_sd, d_posz_sd, d_gridParticleIndex, numCurPar);                                        
			cudaDeviceSynchronize();

			cudaUnbindTexture(posx_d_tex);//data from d_pos_d to d_pos_sd accordingly to sorted index
			cudaUnbindTexture(posy_d_tex);
			cudaUnbindTexture(posz_d_tex);
		    
			cudaUnbindTexture(posx_tex);//data from d_pos to d_pos_s accordingly to sorted index
			cudaUnbindTexture(posy_tex);
			cudaUnbindTexture(posz_tex);
			cudaUnbindTexture(ptype_tex);

			cudaMemset(d_statusPar, 255, sizeof(unsigned char) * iniPar);
			cudaMemset(d_statusPar, 0, sizeof(unsigned char) * numCurPar);
			cudaMemset(d_ptype, 255, sizeof(unsigned char) * iniPar);
			cudaMemcpy(d_ptype, d_ptype_s, sizeof(unsigned char) * numCurPar, cudaMemcpyDeviceToDevice);
//note here, arrays have been sorted for both _s and _sd
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
			
			nblocks = 1 + (numCurPar - 1)/NTHREAD_PER_BLOCK_PAR;
			react4TimeStep_afterDiffusion<<<nblocks, NTHREAD_PER_BLOCK_PAR>>>(d_posx, d_posy, d_posz, d_ptype, d_gridParticleHash, d_idxnzBin_neig, d_idxnzBin_numNeig, d_nzBinidx, d_accumParidxBin, d_statusPar, numBinx, numBiny, numBinz, numNZBin, numCurPar, idx_typedeltaT);	
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
		zip_new_end = thrust::remove_if(zip_begin, zip_end, first_element_equal_255());

		numCurPar = zip_new_end - zip_begin;		
		
		idx_iter++;
		if(idx_iter%100 == 0) printf("idx_iter = %d curTime = %f # of radicals = %d\n", idx_iter, curTime, numCurPar);			
    }
/***********************************************************************************/		
	//revised at Sep 6 2019 by Lai
    cudaMemcpy(h_posx,d_posx,sizeof(float)*numCurPar, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_posy,d_posy,sizeof(float)*numCurPar, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_posz,d_posz,sizeof(float)*numCurPar, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ptype,d_ptype,sizeof(unsigned char)*numCurPar, cudaMemcpyDeviceToHost);
    
    float tmpchem=1-PROBCHEM;
    int numOH=0;
    FILE* fpchem=fopen("./Results/totalchem.dat","ab");
    FILE* fpspecies=fopen("./Results/totalspecpos.dat","ab");
    FILE* fpspetype=fopen("./Results/totalspectype.dat","ab");
	for(int tmptmp=0;tmptmp<numCurPar;tmptmp++)
	{
		fwrite (&h_posx[tmptmp], sizeof(float), 1, fpspecies  );
        fwrite (&h_posy[tmptmp], sizeof(float), 1, fpspecies  );
        fwrite (&h_posz[tmptmp], sizeof(float), 1, fpspecies  );
        fwrite (&h_ptype[tmptmp], sizeof(unsigned char), 1, fpspetype );
		if(h_ptype[tmptmp]==1)
		{
			numOH++;
			fwrite (&h_posx[tmptmp], sizeof(float), 1, fpchem );
            fwrite (&h_posy[tmptmp], sizeof(float), 1, fpchem );
            fwrite (&h_posz[tmptmp], sizeof(float), 1, fpchem );
            fwrite (&tmpchem, sizeof(float), 1, fpchem );
		}
	}		
	fclose(fpchem);
	fclose(fpspecies);
	fclose(fpspetype); //*/



	int totalPar=0,totalOH=0;
	FILE *depofp = NULL;
	depofp = fopen("./Results/chemNum.txt", "r");
	if (depofp == NULL)
	{
		totalPar=0;
		totalOH=0;
	}
	else
	{
		fscanf(depofp, "%d %d", &totalPar,&totalOH);
		fclose(depofp);
	}

    totalPar+=numCurPar;
    totalOH+=numOH;
    depofp = fopen("./Results/chemNum.txt", "w");
    fprintf(depofp, "%d %d", totalPar,totalOH);
    fclose(depofp);//*/

	if(flagDNA)
	{
/***********************************************************************************/
		int* dev_chromatinIndex;
		int* dev_chromatinStart;
		int* dev_chromatinType;
		CoorBasePair* dev_straightChrom;
		CoorBasePair* dev_bendChrom;
		float3* dev_straightHistone;
		float3* dev_bendHistone;
		int totalspace = NUCLEUS_DIM*NUCLEUS_DIM*NUCLEUS_DIM;
		int *chromatinIndex = (int*)malloc(sizeof(int)*totalspace);
		int *chromatinStart = (int*)malloc(sizeof(int)*totalspace);
		int *chromatinType = (int*)malloc(sizeof(int)*totalspace);
		for (int k=0; k<totalspace; k++) 
		{
			chromatinIndex[k] = -1;
			chromatinStart[k] = -1;
			chromatinType[k] = -1;
		}

		int index, data[6];
		long lSize;
		FILE* pFile=fopen("./table/WholeNucleoChromosomesTable.bin","rb");
		fseek (pFile , 0 , SEEK_END);
	    lSize = ftell (pFile);
	  	rewind (pFile);
	  	for (int i=0; i<lSize/(6*sizeof(int)); i++)
		{
		    fread(data,sizeof(int),6, pFile);
		    //if(i<5) printf("%d %d %d %d %d %d\n", data[0], data[1], data[2], data[3], data[4], data[5]);
			index = data[0] + data[1] * NUCLEUS_DIM + data[2] * NUCLEUS_DIM * NUCLEUS_DIM;
			chromatinIndex[index] = data[3];
			chromatinStart[index] = data[4];
			chromatinType[index] = data[5];
		}
		fclose(pFile);

		CUDA_CALL(cudaMalloc((void**)&dev_chromatinIndex, totalspace * sizeof(int)));
		CUDA_CALL(cudaMemcpy(dev_chromatinIndex, chromatinIndex, totalspace * sizeof(int), cudaMemcpyHostToDevice));//DNA index
		CUDA_CALL(cudaMalloc((void**)&dev_chromatinStart, totalspace * sizeof(int)));
		CUDA_CALL(cudaMemcpy(dev_chromatinStart, chromatinStart, totalspace * sizeof(int), cudaMemcpyHostToDevice));//# of start base in the box
		CUDA_CALL(cudaMalloc((void**)&dev_chromatinType, totalspace * sizeof(int)));
		CUDA_CALL(cudaMemcpy(dev_chromatinType, chromatinType, totalspace * sizeof(int), cudaMemcpyHostToDevice));//type of the DNA in the box
	    free(chromatinIndex);
	    free(chromatinStart);
	    free(chromatinType);

		CoorBasePair *StraightChrom = (CoorBasePair*)malloc(sizeof(CoorBasePair)*STRAIGHT_BP_NUM);
		const char *straight = "./table/StraightChromatinFiberUnitTable.txt";
		printf("Straight Chromatin Table: Reading %s\n", straight);
		FILE *fpStraight = fopen(straight,"r");
		int dump;
		float bx, by, bz, rx, ry, rz, lx, ly, lz;
	    for (int i=0; i<STRAIGHT_BP_NUM; i++)
		{
		    fscanf(fpStraight,"%d %f %f %f %f %f %f %f %f %f\n", &dump, &bx, &by, &bz, &rx, &ry, &rz, &lx, &ly, &lz);
		    //if(i<5) printf("%d %f %f %f %f %f %f %f %f %f\n", dump, bx, by, bz, rx, ry, rz, lx, ly, lz);
			StraightChrom[i].base.x = bx;
			StraightChrom[i].base.y = by;
			StraightChrom[i].base.z = bz;
			StraightChrom[i].right.x = rx;
			StraightChrom[i].right.y = ry;
			StraightChrom[i].right.z = rz;
			StraightChrom[i].left.x = lx;
			StraightChrom[i].left.y = ly;
			StraightChrom[i].left.z = lz;
		}
		fclose(fpStraight);
		CUDA_CALL(cudaMalloc((void**)&dev_straightChrom, STRAIGHT_BP_NUM * sizeof(CoorBasePair)));
		CUDA_CALL(cudaMemcpy(dev_straightChrom, StraightChrom, STRAIGHT_BP_NUM * sizeof(CoorBasePair), cudaMemcpyHostToDevice));

		CoorBasePair *BendChrom = (CoorBasePair*)malloc(sizeof(CoorBasePair)*BEND_BP_NUM);
		const char *bend = "./table/BentChromatinFiberUnitTable.txt";
		printf("Bend Chromatin Table: Reading %s\n", bend);
		FILE *fpBend = fopen(bend,"r");
	    for (int i=0; i<BEND_BP_NUM; i++)
		{
		    fscanf(fpBend,"%d %f %f %f %f %f %f %f %f %f\n", &dump, &bx, &by, &bz, &rx, &ry, &rz, &lx, &ly, &lz);
		    //if(i<5) printf("%d %f %f %f %f %f %f %f %f %f\n", dump, bx, by, bz, rx, ry, rz, lx, ly, lz);
			BendChrom[i].base.x = bx;
			BendChrom[i].base.y = by;
			BendChrom[i].base.z = bz;
			BendChrom[i].right.x = rx;
			BendChrom[i].right.y = ry;
			BendChrom[i].right.z = rz;
			BendChrom[i].left.x = lx;
			BendChrom[i].left.y = ly;
			BendChrom[i].left.z = lz;
		}
		fclose(fpBend);
		CUDA_CALL(cudaMalloc((void**)&dev_bendChrom, BEND_BP_NUM * sizeof(CoorBasePair)));
		CUDA_CALL(cudaMemcpy(dev_bendChrom, BendChrom, BEND_BP_NUM * sizeof(CoorBasePair), cudaMemcpyHostToDevice));
		
		float hisx, hisy, hisz;
		float3* bendHistone = (float3*)malloc(sizeof(float3)*BEND_HISTONE_NUM);
		const char *bent = "./table/BentHistonesTable.txt";
		printf("Bent Histone Table: Reading %s\n", bent);
		FILE *fpBentH = fopen(bent,"r");
	    for (int i=0; i<BEND_HISTONE_NUM; i++)
		{
		    fscanf(fpBentH,"%f %f %f\n", &hisx, &hisy, &hisz);
		    //if(i<5) printf("%f %f %f\n", hisx, hisy, hisz);
			bendHistone[i].x = hisx;
			bendHistone[i].y = hisy;
			bendHistone[i].z = hisz;
		}
		fclose(fpBentH);
		CUDA_CALL(cudaMalloc((void**)&dev_bendHistone, BEND_HISTONE_NUM * sizeof(float3)));
		CUDA_CALL(cudaMemcpy(dev_bendHistone, bendHistone, BEND_HISTONE_NUM * sizeof(float3), cudaMemcpyHostToDevice));
		
		float3 *straightHistone = (float3*)malloc(sizeof(float3)*STRAIGHT_HISTONE_NUM);
		const char *straiHistone = "./table/StraightHistonesTable.txt";
		printf("Straight Histone Table: Reading %s\n", straiHistone);
		FILE *fpStraiH = fopen(straiHistone,"r");
	    for (int i=0; i<STRAIGHT_HISTONE_NUM; i++)
		{
		    fscanf(fpStraiH,"%f %f %f\n", &hisx, &hisy, &hisz);
		    //if(i<5) printf("%f %f %f\n", hisx, hisy, hisz);
			straightHistone[i].x = hisx;
			straightHistone[i].y = hisy;
			straightHistone[i].z = hisz;
		}
		fclose(fpStraiH);
		CUDA_CALL(cudaMalloc((void**)&dev_straightHistone, STRAIGHT_HISTONE_NUM * sizeof(float3)));
		CUDA_CALL(cudaMemcpy(dev_straightHistone, straightHistone, STRAIGHT_HISTONE_NUM * sizeof(float3), cudaMemcpyHostToDevice));
		
		free(StraightChrom);
		free(BendChrom);	
		free(bendHistone);	
		free(straightHistone);

		//modelTableSetup(dev_chromatinIndex,dev_chromatinStart,dev_chromatinType,dev_straightChrom,dev_bendChrom,dev_straightHistone,dev_bendHistone);
		printf("DNA geometry has been loaded to GPU memory\n");	 
		int* tmp=(int*) malloc(sizeof(int)*27);
		int kk=0;	
		for(int iz = -1; iz < 2; iz ++)
	    {
	        for(int iy = -1; iy < 2; iy ++)
	        {
		        for(int ix = -1; ix < 2; ix ++)
				{
					tmp[kk] = iz * NUCLEUS_DIM * NUCLEUS_DIM + iy * NUCLEUS_DIM + ix;
					//printf("idx_neig = %d, iz = %d, iy = %d, iz = %d, h_deltaidxBin_neig = %d\n", idx_neig, iz, iy, ix, tmp[idx_neig]);
					kk++;
				}
			}
		}
		CUDA_CALL(cudaMemcpyToSymbol(neighborindex,tmp,sizeof(int)*27,0,cudaMemcpyHostToDevice));
		free(tmp);
		printf("Finish initialize neighborindex\n");

		calDNAreact_radius(rDNA,h_deltaT);//calculate reaction radius
		//for(int i=0;i<6;i++)
		//	printf("radii for OH and DNA components is %f\n", rDNA[i]);
		CUDA_CALL(cudaMemcpyToSymbol(d_rDNA,rDNA,sizeof(float)*6, 0, cudaMemcpyHostToDevice));
/********deal with total physics energy deposit*************************************/
		int totalphy;
		Edeposit* edrop=readStage(&totalphy,0);//read binary file x y z e
		printf("\n**********\ntotal initial number of physics energy deposit point is %d\n**********\n", totalphy);

		Edeposit* dev_edrop;
		cudaMalloc((void**)&dev_edrop,totalphy*sizeof(Edeposit));
		cudaMemcpy(dev_edrop, edrop, totalphy*sizeof(Edeposit), cudaMemcpyHostToDevice);
		free(edrop);
		
		combinePhysics* d_recorde;
		CUDA_CALL(cudaMalloc((void**)&d_recorde,sizeof(combinePhysics)*totalphy));

		

		phySearch<<<(MAXNUMPAR-1)/NTHREAD_PER_BLOCK_PAR+1,NTHREAD_PER_BLOCK_PAR>>>(totalphy, dev_edrop, dev_chromatinIndex,dev_chromatinStart,dev_chromatinType, dev_straightChrom,
									dev_bendChrom, dev_straightHistone, dev_bendHistone, d_recorde);
		cudaDeviceSynchronize();

		combinePhysics* recorde=(combinePhysics*)malloc(sizeof(combinePhysics)*totalphy);		 
		CUDA_CALL(cudaMemcpy(recorde, d_recorde, sizeof(combinePhysics)*totalphy,cudaMemcpyDeviceToHost));

		chemReact* recordPhy= combinePhy(&totalphy, recorde,0);//consider the probability and generate final damage site
		printf("\n**********\neffective physics damage is %d\n**********", totalphy);
		free(recorde);
		CUDA_CALL(cudaFree(d_recorde));
	/**************************************************************/
		int totalchem;
		Edeposit* chemdrop=readStage(&totalchem,1);
		printf("\n**********\ntotal initial number of chemical  point is %d\n**********\n", totalchem);

		Edeposit* dev_chemdrop;
		cudaMalloc((void**)&dev_chemdrop,totalchem*sizeof(Edeposit));
		cudaMemcpy(dev_chemdrop, chemdrop, totalchem*sizeof(Edeposit), cudaMemcpyHostToDevice);
		free(chemdrop);
		combinePhysics* d_recordc;
		CUDA_CALL(cudaMalloc((void**)&d_recordc,sizeof(combinePhysics)*totalchem));

		chemSearch<<<(MAXNUMPAR-1)/NTHREAD_PER_BLOCK_PAR+1,NTHREAD_PER_BLOCK_PAR>>>(totalchem, dev_chemdrop, dev_chromatinIndex,dev_chromatinStart,dev_chromatinType, dev_straightChrom,
									dev_bendChrom, dev_straightHistone, dev_bendHistone, d_recordc);
		cudaDeviceSynchronize();

		combinePhysics* recordc=(combinePhysics*)malloc(sizeof(combinePhysics)*totalchem);		 
		CUDA_CALL(cudaMemcpy(recordc, d_recordc, sizeof(combinePhysics)*totalchem,cudaMemcpyDeviceToHost));

		chemReact* recordChem= combinePhy(&totalchem, recordc,1);//consider the probability and generate final damage site
		printf("\n**********\neffective chemical damage is %d\n**********", totalchem);
		free(recordc);
		CUDA_CALL(cudaFree(d_recordc));

		if(totalphy+totalchem==0) {printf("NO DAMAGE AT ALL\n");return;}

	    chemReact* totalrecord=(chemReact*)malloc(sizeof(chemReact)*(totalphy+totalchem));
	    memcpy(totalrecord,recordPhy,sizeof(chemReact)*totalphy);
	    memcpy(&totalrecord[totalphy],recordChem,sizeof(chemReact)*totalchem);
	    free(recordPhy);
	    free(recordChem);	    
	    printf("total efective is %d\n**********", totalphy+totalchem);

	    damageAnalysis(totalphy+totalchem,totalrecord);
	    free(totalrecord);//*/
	    CUDA_CALL(cudaFree(dev_chromatinIndex));	
		CUDA_CALL(cudaFree(dev_chromatinType));
		CUDA_CALL(cudaFree(dev_chromatinStart));
		CUDA_CALL(cudaFree(dev_straightHistone));
		CUDA_CALL(cudaFree(dev_straightChrom));
		CUDA_CALL(cudaFree(dev_bendHistone));
		CUDA_CALL(cudaFree(dev_bendChrom));
	}	
	uniBinidxPar_dev_vec.clear();	
}