#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <string.h>
#include <vector>
#include <pthread.h>
#include <time.h>
using namespace std;


#include "microMC_chem.h"
#include "initialization.cu"
#include "runMicroMC.cu"
#include "libGeometryRand.cu"
#include "microMC_kernels.cu"
//#include "realtime.cu"

int main(int argc, char* argv[])
{
	cudaDeviceReset();
	
	time_t start_time, end_time, end_time_ini, end_time_sim;
	float time_total, time_ini, time_sim;
	int deviceNo, process_time, flagDNA;
	start_time = clock();
	
	if (argc != 4)
	{
		printf("Please execute ./chem GPUdeviceNo process_time(ps) flag\n");
		printf("Thanks.\n\n");
		exit(1);
	}

	deviceNo = atoi(argv[1]);
	cudaSetDevice(deviceNo);
	cudaDeviceReset();
	//printDevProp(deviceNo);

	process_time = atoi(argv[2]);
	flagDNA = atoi(argv[3]);

	ChemistrySpec chemData;
	chemData.initChemistry("./Input/chemistryData");

	ReactionType reactType1;
	reactType1.initReaction(chemData, "./Input/chemistryData/Type1");

	ParticleData parData;
	parData.readInitialParticles_GEANT4("../prechem_stage/output_afterremove.bin");

	initGPUVariables(&chemData, &reactType1, &parData);
	
	end_time_ini = clock();
	
	runMicroMC(&chemData, &reactType1, &parData, process_time, flagDNA);

	end_time_sim = clock();
	
	cudaFree(d_posx);
	cudaFree(d_posy);
    cudaFree(d_posz);
	cudaFree(d_ptype);
	
	cudaFree(d_posx_s);
	cudaFree(d_posy_s);
    cudaFree(d_posz_s);
	cudaFree(d_ptype_s);
	
	cudaFree(d_posx_d);
	cudaFree(d_posy_d);
    cudaFree(d_posz_d);

	cudaFree(d_posx_sd);
	cudaFree(d_posy_sd);
    cudaFree(d_posz_sd);

	cudaFree(d_statusPar);     
	cudaFree(d_gridParticleHash); 
	cudaFree(d_gridParticleIndex); 
    cudaFree(d_accumParidxBin);
	//cudaFree(d_numParBinidx);
	cudaFree(d_nzBinidx);
	cudaFree(d_idxnzBin_neig);
	
	cudaFree(d_mintd_Par);
	
	//cudaFree(d_tagnzBin);
	
	end_time = clock();
	
	time_ini = ((float)end_time_ini - (float)start_time) / CLOCKS_PER_SEC;
	time_sim = ((float)end_time_sim - (float)end_time_ini) / CLOCKS_PER_SEC;
	time_total = ((float)end_time - (float)start_time) / CLOCKS_PER_SEC;
	
	printf("\n\n****************************************\n");
	printf("Total computation time: %f seconds.\n\n", time_total);
	printf("Init time: %f seconds.\n\n", time_ini);
	printf("Total simulation time: %f seconds.\n\n", time_sim);
	printf("****************************************\n\n\n");
	
	return 0;
}
