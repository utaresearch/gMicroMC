#include "global.h"
#include "initialize.h"
#include "globalKernel.cuh"

void inirngG(int value)
{
//  initialize rand seeds at CPU
    printf("\nStart initialize random numbers\n");
    if(value == 0) srand( (unsigned int)time(NULL) );
    else  srand ( value );

    int* iseed1_h = (int*) malloc(sizeof(int)*NRAND);
    if(iseed1_h == NULL) printf("MALLOC error\n");
//  generate randseed at CPU
    for(int i = 0; i < NRAND; i++)
    {
        iseed1_h[i] = rand();
    }
    int *iseed1;
    CUDA_CALL(cudaMalloc((void **) &iseed1,sizeof(int)*NRAND));
    CUDA_CALL(cudaMemcpy(iseed1, iseed1_h, sizeof(int)*NRAND, cudaMemcpyHostToDevice));
    free(iseed1_h);
    
    iniCuseed(iseed1);
    cudaFree(iseed1);
}

void initialize(std::string ss)
{
    document.Parse<kParseCommentsFlag>(ss.c_str());
	verbose = document["verbose"].GetInt();	
	NPART = document["NPART"].GetInt();
	NRAND = document["NRAND"].GetInt();
	NTHREAD_PER_BLOCK = document["NTHREAD_PER_BLOCK"].GetInt();
	deviceIndex = document["Device"].GetInt();

	std::cout << "verbose is "<< verbose <<std::endl;
	std::cout << "NPART is "<< NPART <<std::endl;
	std::cout << "NRAND is "<< NRAND <<std::endl;
	std::cout << "NTHREAD_PER_BLOCK is "<< NTHREAD_PER_BLOCK <<std::endl;

    std::cout<<"trying to use device "<< deviceIndex <<std::endl;
	if(cudaSetDevice(document["Device"].GetInt())==cudaSuccess)
		printf("set device %d success\n", document["Device"].GetInt());
    
    cudaGetDeviceProperties(&devProp, deviceIndex);
    if(verbose>0)
        printDevProp(devProp);

    // initialize seeds for getting random numbers
    inirngG(0);
    //
    copyConstantToGPU();
}
void copyConstantToGPU()
{

}

void printDevProp(cudaDeviceProp devProp)
//      print out device properties
{
    int devCount;
    cudaGetDeviceCount(&devCount);
	std::cout << "Number of device:              " << devCount << std::endl;
	std::cout << "Using device #:                " << deviceIndex << std::endl;

//      device properties	
	printf("Major revision number:         %d\n",  devProp.major);
    printf("Minor revision number:         %d\n",  devProp.minor);
    printf("Name:                          %s\n",  devProp.name);
    printf("Total global memory:           %7.2f MB\n",  
	devProp.totalGlobalMem/1024.0/1024.0);
    printf("Total shared memory per block: %5.2f kB\n",  
	devProp.sharedMemPerBlock/1024.0);
    printf("Total registers per block:     %u\n",  devProp.regsPerBlock);
    printf("Warp size:                     %d\n",  devProp.warpSize);
    printf("Maximum memory pitch:          %lu\n",  devProp.memPitch);
    printf("Maximum threads per block:     %d\n",  devProp.maxThreadsPerBlock);
    	
	printf("Maximum dimension of block:    %d*%d*%d\n", 			
	devProp.maxThreadsDim[0],devProp.maxThreadsDim[1],devProp.maxThreadsDim[2]);
	printf("Maximum dimension of grid:     %d*%d*%d\n", 
	devProp.maxGridSize[0],devProp.maxGridSize[1],devProp.maxGridSize[2]);
    printf("Clock rate:                    %4.2f GHz\n",  devProp.clockRate/1000000.0);
    printf("Total constant memory:         %5.2f kB\n",  devProp.totalConstMem/1024.0);
    printf("Texture alignment:             %lu\n",  devProp.textureAlignment);
    printf("Concurrent copy and execution: %s\n",  (devProp.deviceOverlap ? "Yes" : "No"));
    printf("Number of multiprocessors:     %d\n",  devProp.multiProcessorCount);
    printf("Kernel execution timeout:      %s\n",  (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
//      obtain computing resource

}