#include <stdlib.h>
#include <string.h>
#include <iostream>
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

#include <cusparse.h>

#include "microMC_prechem_global.h"

struct first_element_equal_255
{
  __host__ __device__
  bool operator()(const thrust::tuple<const int&, const float&, const float&, const float&, const int&, const float&> &t)
  {
      return thrust::get<0>(t) == 255;
  }
};

void runMicroMC_pc(ParticleData_prechem *parData_pc, Branch_water_prechem *braInfo_pc, ThermRecomb_elec_prechem *thermRecombInfo_pc)
{
    cudaStream_t stream[5];
	for(int i=0; i<5; i++)
	    cudaStreamCreate(&stream[i]);

	//simulating the prechemical stage for the subexcitation electrons: thermalisation or recombination with its parent ionized water
    int nblocks = 1 + (parData_pc->num_elec - 1)/NTHREAD_PER_BLOCK_PAR;
	thermalisation_subexelectrons<<<nblocks,NTHREAD_PER_BLOCK_PAR, 0, stream[0]>>>(d_posx, d_posy, d_posz, d_ene, d_ptype, d_statetype, d_wiid_elec);
	cudaStreamSynchronize(stream[0]);
	
	//simulating the prechemical stage for the ionized water (the ones don't have recombination with water
	nblocks = 1 + (parData_pc->num_wi - 1)/NTHREAD_PER_BLOCK_PAR;
	dissociation_ionizedwater<<<nblocks, NTHREAD_PER_BLOCK_PAR, 0, stream[0]>>>(d_posx, d_posy, d_posz, d_ptype, d_statetype); 
	
	//simulating the prechemical stage for the excited water with A1B1 excitation state
	nblocks = 1 + (parData_pc->num_we_a1b1 - 1)/NTHREAD_PER_BLOCK_PAR;
	dissociation_excitedwater_a1b1<<<nblocks, NTHREAD_PER_BLOCK_PAR, 0, stream[1]>>>(d_posx, d_posy, d_posz, d_ptype, d_statetype); 
	
	//simulating the prechemical stage for the excited water with A1B1 excitation state
	nblocks = 1 + (parData_pc->num_we_b1a1 - 1)/NTHREAD_PER_BLOCK_PAR;
	dissociation_excitedwater_b1a1<<<nblocks, NTHREAD_PER_BLOCK_PAR, 0, stream[2]>>>(d_posx, d_posy, d_posz, d_ptype, d_statetype); 
	
	//simulating the prechemical stage for the excited water with A1B1 excitation state
	nblocks = 1 + (parData_pc->num_we_rd - 1)/NTHREAD_PER_BLOCK_PAR;
	dissociation_excitedwater_rd<<<nblocks, NTHREAD_PER_BLOCK_PAR, 0, stream[3]>>>(d_posx, d_posy, d_posz, d_ptype, d_statetype); 
	
	//simulating the prechemical stage for the excited water with A1B1 excitation state
	nblocks = 1 + (parData_pc->num_w_dis - 1)/NTHREAD_PER_BLOCK_PAR;
	dissociation_dissociativewater<<<nblocks, NTHREAD_PER_BLOCK_PAR, 0, stream[4]>>>(d_posx, d_posy, d_posz, d_ptype, d_statetype);
	
	cudaDeviceSynchronize();
	FILE *fp;
	float *output_posx, *output_posy, *output_posz, *output_ttime;
    int *output_ptype, *output_index;
	// output the result without removing 
	/*float *output_posx = (float*)malloc(sizeof(float) * parData_pc->num_total * 3);
    cudaMemcpyAsync(output_posx , d_posx, sizeof(float)*parData_pc->num_total * 3, cudaMemcpyDeviceToHost, stream[0]);	
	
	float *output_posy = (float*)malloc(sizeof(float) * parData_pc->num_total * 3);
    cudaMemcpyAsync(output_posy , d_posy, sizeof(float)*parData_pc->num_total * 3, cudaMemcpyDeviceToHost, stream[1]);	
	
	float *output_posz = (float*)malloc(sizeof(float) * parData_pc->num_total * 3);
    cudaMemcpyAsync(output_posz , d_posz, sizeof(float)*parData_pc->num_total * 3, cudaMemcpyDeviceToHost, stream[2]);

    int *output_ptype = (int*)malloc(sizeof(int) * parData_pc->num_total * 3);
    cudaMemcpyAsync(output_ptype, d_ptype, sizeof(int)*parData_pc->num_total * 3, cudaMemcpyDeviceToHost, stream[3]);	

    int *output_statetype = (int*)malloc(sizeof(int) * parData_pc->num_total);
    cudaMemcpyAsync(output_statetype , d_statetype, sizeof(int)*parData_pc->num_total, cudaMemcpyDeviceToHost, stream[4]);		
	
	cudaDeviceSynchronize();
	
	fp = fopen("output.bin", "wb");	
    fwrite(output_posx, sizeof(float), parData_pc->num_total * 3, fp);
    fwrite(output_posy, sizeof(float), parData_pc->num_total * 3, fp);
	fwrite(output_posz, sizeof(float), parData_pc->num_total * 3, fp);
	fwrite(output_ptype, sizeof(int), parData_pc->num_total * 3, fp);
	fwrite(output_statetype, sizeof(int), parData_pc->num_total, fp);
	fclose(fp);//*/
	
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
	zip_end   = zip_begin + parData_pc->num_total * 3;  		
	zip_new_end = thrust::remove_if(zip_begin, zip_end, first_element_equal_255());
	
	cudaDeviceSynchronize();
	
	int	numCurPar = zip_new_end - zip_begin;
		
	printf("After removing, numCurPar = %d\n", numCurPar);
	output_posx = (float*)malloc(sizeof(float) * numCurPar);
    output_posy = (float*)malloc(sizeof(float) * numCurPar);
    output_posz = (float*)malloc(sizeof(float) * numCurPar);
    output_ttime = (float*)malloc(sizeof(float) * numCurPar);
    output_ptype = (int*)malloc(sizeof(float) * numCurPar);
    output_index = (int*)malloc(sizeof(float) * numCurPar);
    
    cudaMemcpyAsync(output_posx , d_posx, sizeof(float)*numCurPar, cudaMemcpyDeviceToHost, stream[0]);	
    cudaMemcpyAsync(output_posy , d_posy, sizeof(float)*numCurPar, cudaMemcpyDeviceToHost, stream[1]);	
    cudaMemcpyAsync(output_posz , d_posz, sizeof(float)*numCurPar, cudaMemcpyDeviceToHost, stream[2]);
    cudaMemcpyAsync(output_ptype, d_ptype, sizeof(int)*numCurPar, cudaMemcpyDeviceToHost, stream[3]);	
    cudaMemcpyAsync(output_index, d_index, sizeof(int)*numCurPar, cudaMemcpyDeviceToHost, stream[4]);	
    cudaMemcpyAsync(output_ttime, d_ttime, sizeof(int)*numCurPar, cudaMemcpyDeviceToHost, stream[4]);	

	cudaDeviceSynchronize();
	
	fp = fopen("output_afterremove.bin", "w");	
    fwrite(output_posx, sizeof(float), numCurPar, fp);
    fwrite(output_posy, sizeof(float), numCurPar, fp);
	fwrite(output_posz, sizeof(float), numCurPar, fp);
	fwrite(output_ttime, sizeof(float), numCurPar, fp);
	fwrite(output_index, sizeof(int), numCurPar, fp);
	fwrite(output_ptype, sizeof(int), numCurPar, fp);
	fclose(fp);	
	
	/*fp = fopen("totalspec0.dat", "ab");	
    fwrite(output_posx, sizeof(float), numCurPar, fp);
    fwrite(output_posy, sizeof(float), numCurPar, fp);
	fwrite(output_posz, sizeof(float), numCurPar, fp);
	fwrite(output_ttime, sizeof(float), numCurPar, fp);
	fwrite(output_index, sizeof(int), numCurPar, fp);
	fwrite(output_ptype, sizeof(int), numCurPar, fp);
	fclose(fp);	//*/

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

