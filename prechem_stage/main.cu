#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <string.h>
#include <vector>
#include <pthread.h>
#include <time.h>
using namespace std;


#include "microMC_prechem_global.h"
#include "initialization.cu"
#include "libGeometryRand.cu"
#include "microMC_pc_kernels.cu"
#include "runMicroMC_pc.cu"

int deviceNo;

int main(int argc, char* argv[])
{
	cudaDeviceReset();
	
	time_t start_time, end_time, end_time_ini;
	float time_total, time_ini;

	start_time = clock();
	
	if (argc != 2)
	{
		printf("Please execute ./prechem GPUdeviceNo\n");
		printf("Thanks.\n\n");
		exit(1);
	}

	deviceNo = atoi(argv[1]);
	cudaSetDevice(deviceNo);
		
	ParticleData_prechem parData_pc;
	parData_pc.readInitialParticles_GEANT4();
	
	Branch_water_prechem braInfo_pc;
	braInfo_pc.readBranchInfo("./Input/branchInfo_prechem_org.txt");
	
	ThermRecomb_elec_prechem thermRecombInfo_pc;
	thermRecombInfo_pc.readThermRecombInfo("./Input/thermRecombInfo_prechem.txt");
	
	initGPUVariables_pc(&parData_pc, &braInfo_pc, &thermRecombInfo_pc);
	end_time_ini = clock();

	runMicroMC_pc(&parData_pc, &braInfo_pc, &thermRecombInfo_pc);
	end_time = clock();
	
	time_ini = ((float)end_time_ini - (float)start_time) / CLOCKS_PER_SEC;
	time_total = ((float)end_time - (float)start_time) / CLOCKS_PER_SEC;
	
	printf("\n\n****************************************\n");
	printf("Init time: %f seconds.\n\n", time_ini);
	printf("Total computation time: %f seconds.\n\n", time_total);
	printf("****************************************\n\n\n");
	
	return 0;
}
