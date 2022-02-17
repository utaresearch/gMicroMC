
#include "physicsList.h"
#include "physicsKernel.cuh"
#include "globalKernel.cuh"

PhysicsList::PhysicsList()
// initialize physics list, read neccessary data for physical transport
{
	std::cout << "Initializing Physical Stage" << std::endl;
	np = document["nPar"].GetInt();;
	h_particles = new Particle[np];

	channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	float ecut = document["eECutoff"].GetFloat();
	CUDA_CALL(cudaMemcpyToSymbol(&eEcutoff, &ecut, sizeof(float)));
	ecut = document["pECutoff"].GetFloat();
	CUDA_CALL(cudaMemcpyToSymbol(&pECutoff, &ecut, sizeof(float)));

	int shape = document["physicsWorldShape"].GetInt();
	CUDA_CALL(cudaMemcpyToSymbol(&boundaryShape, &shape, sizeof(int)));
	float3 center = {0};
	center.x = document["physicsWorldCenterX"].GetFloat();
	center.y = document["physicsWorldCenterY"].GetFloat();
	center.z = document["physicsWorldCenterZ"].GetFloat();
	CUDA_CALL(cudaMemcpyToSymbol(&boundaryCenter, &center, sizeof(float3)));
	float3 size = {0};
	size.x = document["physicsWorldSizeX"].GetFloat();
	size.y = document["physicsWorldSizeY"].GetFloat();
	size.z = document["physicsWorldSizeZ"].GetFloat();
	CUDA_CALL(cudaMemcpyToSymbol(&boundarySize, &size, sizeof(float3)));

	printf("\n\n\nTest world %d %f %f %f %f %f %f\n\n\n", shape, center.x, center.y, center.z, size.x, size.y, size.z);
	readDataToGPU();
}
void PhysicsList::iniParticle()
// generating particles according to the given model
{
	int sourceModel = document["sourceModel"].GetInt();
	if(sourceModel!=0 && sourceModel!=1)	
	{
		std::cout << "Source Model Not Supported Now!!!" << std::endl;
		exit(EXIT_FAILURE);
	}

	if(sourceModel==0) // read from PSF
	{
		float inx, iny, inz, dirx, diry, dirz, e, uniform, A;
		int ptype;  
		std::string filename = document["sourceFile"].GetString();
		FILE *fp = fopen(filename.c_str(),"r");
		if(!fp) 
		{
			fprintf(stderr, "Failed to open config file %s\n", filename.c_str());
			exit(EXIT_FAILURE);
		} 
		for(int i = 0; i < np; i++)
		{
			fscanf(fp, "%d %f %f %f %f %f %f %f %f\n",&ptype, &A, &inx, &iny, &inz, &dirx, &diry, &dirz, &e);
			h_particles[i].ptype = ptype;
			h_particles[i].A = A;
			h_particles[i].x = inx;
			h_particles[i].y = iny;
			h_particles[i].z = inz;
			uniform = sqrt(dirx*dirx+diry*diry+dirz*dirz);        
			h_particles[i].ux = dirx/uniform;
			h_particles[i].uy = diry/uniform;
			h_particles[i].uz = dirz/uniform;        
			h_particles[i].e = e;

			h_particles[i].h2oState = 99;
			h_particles[i].dead = 0;
			h_particles[i].path = 0;     
			h_particles[i].elape_time = 0.0;     
			h_particles[i].id = i;     
			h_particles[i].parentId = -1;
		}
		fclose(fp);
		return;
	}
	if(sourceModel==1)
	{
		int ptype = document["sourcePType"].GetInt();
		float A = document["sourceA"].GetFloat();
		float R = document["sourceSampleDim"].GetFloat();

		int sourceEModel = document["sourceEnergyModel"].GetInt();
		float *h_eneprob = NULL;
		int numBins = 0;
		if(sourceEModel==1) // read in histogram for energy spectrum
		{
			std::string filename = document["sourceFile"].GetString();
			FILE *fp = fopen(filename.c_str(),"r");
			if(!fp) 
			{
				fprintf(stderr, "Failed to open config file %s\n", filename.c_str());
				exit(EXIT_FAILURE);
			}
			else
			{
				float BinStart=0, deltaBin = 1000;
				fscanf(fp, "%d %f %f\n",&numBins, &BinStart, &deltaBin);
				h_eneprob = new float[numBins+3];
				h_eneprob[0] = numBins;
				h_eneprob[1] = BinStart;
				h_eneprob[2] = deltaBin;
				float maxP = 0;
				for(int i=0;i<numBins;i++)
				{
					fscanf(fp, "%f\n", &(h_eneprob[i+3]));
					if(h_eneprob[i+3]> maxP) maxP = h_eneprob[i+3];
				}
				fclose(fp);
				if(maxP>0)
					for(int i=0;i<numBins;i++)
						h_eneprob[i+3] /= maxP; // normalize to 1
				else
				{
					std::cout << "WRONG PROBABILITY!!!!" << std::endl;
					exit(EXIT_FAILURE);
				}	
			}
		}
		else
		{
			float emin = document["sourceEmin"].GetFloat();
			float emax = document["sourceEmax"].GetFloat();
			numBins = 11;
			float BinStart = emin, deltaBin = (emax-emin)*0.1;
			h_eneprob = new float[14];
			h_eneprob[0] = numBins;
			h_eneprob[1] = BinStart;
			h_eneprob[2] = deltaBin;
			for(int i=0;i<numBins;i++)
					h_eneprob[i+3] = 1.0;
		}
		
		sampleSource(np, ptype, A, R, h_eneprob, h_particles);
	}
	max_e = 0;
	nWaitElec = 0;	
	for(int i =0; i<np; i++)  // This part can be done in GPU, but consider typically the numebr of primary particles is less than one million
	// no need to bother with GPU
	{
		if(h_particles[i].e>max_e) max_e = h_particles[i].e;
		if(h_particles[i].ptype == -1) nWaitElec += 1;
	}
}
void PhysicsList::outputParticle()
{
/* save sampled or read-from-PSF particles to file*/
	FILE* fp=fopen("./output/totalsource.txt","a");
    for(int i = 0; i < np; i++)
    {
        fprintf(fp, "%d %.9f %.9f %.9f %.5f %.5f %.5f %.1e\n", h_particles[i].ptype, h_particles[i].x,h_particles[i].y,h_particles[i].z,h_particles[i].ux,
		h_particles[i].uy,h_particles[i].uz,h_particles[i].e);
    }
    fclose(fp);
    printf("finish writing source position\n");
}
PhysicsList::~PhysicsList()
{
	delete[] h_particles;

	CUDA_CALL(cudaDestroyTextureObject(texObj_protonTable)); 
	CUDA_CALL(cudaDestroyTextureObject(texObj_DACSTable));
	CUDA_CALL(cudaDestroyTextureObject(texObj_BindE_array));
	CUDA_CALL(cudaDestroyTextureObject(texObj_ieeCSTable));
	CUDA_CALL(cudaDestroyTextureObject(texObj_elastDCSTable));

	CUDA_CALL(cudaFreeArray(dev_protonTable));
	CUDA_CALL(cudaFreeArray(dev_DACSTable));
    CUDA_CALL(cudaFreeArray(dev_BindE_array));
    CUDA_CALL(cudaFreeArray(dev_ieeCSTable));
    CUDA_CALL(cudaFreeArray(dev_elastDCSTable));

	CUDA_CALL(cudaFree(dev_pQueue));
	CUDA_CALL(cudaFree(dev_eQueue));
	CUDA_CALL(cudaFree(dev_e2Queue));
    CUDA_CALL(cudaFree(dev_container));
    CUDA_CALL(cudaFree(dev_where));
    CUDA_CALL(cudaFree(dev_second_num));
	free(h_container);
}

void PhysicsList::run()
{
	// generating or reading in particles
	iniParticle();
	outputParticle();

	// This part of the code finds the number of primary particle per batch and estimated number of secondary 
	// electrons and waterradiolysis events
	int globalMB = devProp.totalGlobalMem >> 20;    
    int scale = 1 + max_e/100e3;
    ContainerSize = np * scale * document["nRadicalsPer100keV"].GetInt();
    MaxN = np * scale * document["nSecondPer100keV"].GetInt();
	int memory_rqMB =  ContainerSize*sizeof(Data) + MaxN*sizeof(Particle);
	memory_rqMB = document["GPUBaseMemory"].GetInt() + memory_rqMB >> 20;

    int batch = 1 + memory_rqMB/globalMB;   
    enumPerBatch = np/batch; 
	ContainerSize = enumPerBatch * scale * document["nRadicalsPer100keV"].GetInt();
    MaxN = enumPerBatch * scale * document["nSecondPer100keV"].GetInt();
	

	if(verbose > 0)
	{
		printf("Total memory %d MB, estimated required memory %d MB", globalMB, memory_rqMB);
		printf("Total incident particles = %d in batchs = %d\n", np, batch);
		printf("particles per batch is %d\n", enumPerBatch);
		printf("Estimated Max. batch radical states = %d\n", ContainerSize);
		printf("Estimated Max. batch 2nd particles = %d\n", MaxN);
	}

	if(nWaitElec != np)
		thrust::sort(thrust::host, h_particles, h_particles + np, sortPtypeDescend<Particle>());

	// for(int i = 0; i < np; i++)
    // {
    //     printf("%d %.1e\n", h_particles[i].ptype,h_particles[i].e);
    // }

	// runPrint(texObj_ieeCSTable);
	
	nElecPrimary = 131072;
    cudaMalloc(&dev_pQueue, enumPerBatch * sizeof(Particle));
	cudaMalloc(&dev_eQueue, nElecPrimary * sizeof(Particle));
    cudaMalloc(&dev_e2Queue, MaxN * sizeof(Particle));
    
    cudaMalloc(&dev_container, ContainerSize * sizeof(Data));
    h_container = (Data *)malloc(ContainerSize * sizeof(Data));

    cudaMalloc(&dev_where, sizeof(int));
    cudaMalloc(&dev_second_num, sizeof(int));
    cudaMalloc(&dev_gEid, sizeof(int));

	where_all = 0; // variables that will be called in other functions
	sim_num = 0;
	totalTime = 0;

	int gEid = 0;   
    long long where = 0;
    int second_num = 0;    
	int nSim = 0;
	
	system("rm ./output/events.dat");
    for (int i = 0; i<batch+1; i++)
    {
		if(nSim < np - nWaitElec)
		{
			if(nSim + enumPerBatch < np - nWaitElec)
				sim_num = enumPerBatch;
			else 
				sim_num = np - nWaitElec - nSim;
			physicsPType = 1;
		}
		else
		{	
			if(nSim + enumPerBatch < np )
				sim_num = enumPerBatch;
			else 
				sim_num = np - nSim;
			physicsPType = -1;
		}
		
        if(sim_num < 1) break;
        printf("sim_num is %d\n", sim_num);

        CUDA_CALL(cudaMemcpy(dev_where, &where, sizeof(int), cudaMemcpyHostToDevice));//initialize number
        CUDA_CALL(cudaMemcpy(dev_gEid, &gEid, sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(dev_second_num, &second_num, sizeof(int), cudaMemcpyHostToDevice));

		if(physicsPType==1)
        	CUDA_CALL(cudaMemcpy(dev_pQueue, &h_particles[nSim], sim_num*sizeof(Particle), cudaMemcpyHostToDevice));
		else
			CUDA_CALL(cudaMemcpy(dev_eQueue, &h_particles[nSim], sim_num*sizeof(Particle), cudaMemcpyHostToDevice));

		printf("physicsPType is %d\n", physicsPType);
		sortEofElectron(dev_eQueue, sim_num); // can be deleted

        transportParticles();
		nSim += sim_num;

        CUDA_CALL(cudaMemcpy(&second_num, dev_second_num, sizeof(int), cudaMemcpyDeviceToHost));
        printf("second_num is %d\n", second_num);          
        sim_num = second_num;

        cudaMemcpy(&where, dev_where, sizeof(int), cudaMemcpyDeviceToHost);
        CUDA_CALL(cudaMemcpy(h_container, dev_container, where * sizeof(Data), cudaMemcpyDeviceToHost));
        where_all += where;
		printf("where whereall are %lld %lld\n", where, where_all);

        FILE *report = fopen("./output/events.dat","ab");
        fwrite(h_container, where * sizeof(Data), 1, report);
        fclose(report);

        where = 0; // clear container array
        second_num = 0; // clear secondary elctron array     
        /*******************************electron transport*************/
        while(sim_num>0)
        {
			sortEofElectron(dev_e2Queue, sim_num);

			if(nElecPrimary<sim_num)
			{
				cudaFree(dev_eQueue);
				printf("changing eQueue size\n");
				CUDA_CALL(cudaMalloc(&dev_eQueue, sim_num* sizeof(Particle))); // in case memory is not enough. Free original and then assign new memory
				nElecPrimary = sim_num;
			}           
            CUDA_CALL(cudaMemcpy(dev_eQueue, dev_e2Queue, sim_num* sizeof(Particle), cudaMemcpyDeviceToDevice));// put secondary electron as primary

            CUDA_CALL(cudaMemcpy(dev_where, &where, sizeof(int), cudaMemcpyHostToDevice));//initialize number, clear container and secondary array
            CUDA_CALL(cudaMemcpy(dev_second_num, &second_num, sizeof(int), cudaMemcpyHostToDevice));

			physicsPType = -1;
			transportParticles();

            CUDA_CALL(cudaMemcpy(&second_num, dev_second_num, sizeof(int), cudaMemcpyDeviceToHost));
            printf("second_num is %d\n", second_num);          
            sim_num = second_num;    
            
            cudaMemcpy(&where, dev_where, sizeof(int), cudaMemcpyDeviceToHost);
            CUDA_CALL(cudaMemcpy(h_container, dev_container, where * sizeof(Data), cudaMemcpyDeviceToHost));
            where_all += where;
			printf("where whereall are %lld %lld\n", where, where_all);

            report = fopen("./output/events.dat","ab");
            fwrite(h_container, where * sizeof(Data), 1, report);
            fclose(report);
          
            where = 0;
            second_num = 0;
        }                   
    }
	printf("total Time is %f ms\n", totalTime);
}

void PhysicsList::saveResults()
{
	FILE *report = fopen("./output/events.dat","rb");
    int start = ftell(report);
    fseek(report,0,SEEK_END);
    int end = ftell(report);
    int nRows = (end-start)/sizeof(Data);
    if(nRows > ContainerSize)
    {
        free(h_container);
        h_container = (Data*) malloc(nRows*sizeof(Data));
    }
    if(nRows != where_all) printf("problem in reading !!! Consider cleaning ./output first\n");
    fseek(report,0,SEEK_SET);
    fread(h_container, sizeof(Data)*nRows, 1,report);
    fclose(report);

	int state[13] = {0};
    float x,y,z,r2,e;
    float deposit_e = 0.0,total_e=0.0,ecutoff=0;
	std::string fname = document["fileForTotalEvent"].GetString();
    FILE* totalphy=fopen(fname.c_str(),"ab");  
	fname = document["fileForIntOutput"].GetString(); 
    FILE* physint=fopen(fname.c_str(),"wb");
	fname = document["fileForFloatOutput"].GetString();
    FILE* physfloat=fopen(fname.c_str(),"wb");
    int outphyint=0;
   
   	float tmpTotal = 0;
   	int shapeROI = document["ROIShape"].GetInt();
	float3 centerROI = {0};
	centerROI.x = document["ROICenterX"].GetFloat();
	centerROI.y = document["ROICenterY"].GetFloat();
	centerROI.z = document["ROICenterZ"].GetFloat();

	float3 sizeROI = {0};
	sizeROI.x = document["ROISizeX"].GetFloat();
	sizeROI.y = document["ROISizeY"].GetFloat();
	sizeROI.z = document["ROISizeZ"].GetFloat();
	
   	printf("\n ROI shape and size %d %f %f %f\n", shapeROI, sizeROI.x,sizeROI.y,sizeROI.z);
    for (long long i=0; i<where_all; i++)
    {
        x = 1e7*h_container[i].x;
        y = 1e7*h_container[i].y;
        z = 1e7*h_container[i].z;
        e = h_container[i].e;

        tmpTotal += h_container[i].e;
        r2 = x*x+y*y+z*z;
        //if(r2>1e6) continue;
        fwrite(&(h_container[i].parentId),sizeof(int),1,physint);
        fwrite(&(h_container[i].id),sizeof(int),1,physint);
        
        if (h_container[i].h2oState == -1)
        { // electron
            outphyint=0;
            state[12]++;
            ecutoff += h_container[i].e;
        }
        else // water related state
        {
            outphyint=7;
            state[h_container[i].h2oState]++;
        }
        fwrite(&outphyint,sizeof(int),1,physint);
        
        if (h_container[i].h2oState == 11)   // DA changes to 10 for prechem read-in.
            outphyint=10;
        else
            outphyint=h_container[i].h2oState;
        fwrite(&outphyint,sizeof(int),1,physint);
        //fwrite(&h_container[i].h2oState,sizeof(int),1,physint);

        total_e += h_container[i].e;
        
        if (applyROISearch(shapeROI,centerROI, sizeROI, x, y, z))
//        	printf("\n %f %f %f \n", x, y, z);
        {
            deposit_e += h_container[i].e;
            fwrite (&x, sizeof(float), 1, totalphy);
            fwrite (&y, sizeof(float), 1, totalphy);
            fwrite (&z, sizeof(float), 1, totalphy);
            fwrite (&(h_container[i].e), sizeof(float), 1, totalphy);
        }
        fwrite(&(h_container[i].e),sizeof(float),1,physfloat);
        fwrite(&(x),sizeof(float),1,physfloat);         
        fwrite(&(y),sizeof(float),1,physfloat);       
        fwrite(&(z),sizeof(float),1,physfloat);        
        fwrite(&(h_container[i].time),sizeof(float),1,physfloat);
    }

	printf("\nIn total\n");
    outphyint=state[12];
    printf("elec %d\n", outphyint);
    fwrite(&outphyint,sizeof(int),1,physint);
    outphyint=state[0]+state[1]+state[2]+state[3]+state[4];
    printf("ionize %d\n", outphyint);
    fwrite(&outphyint,sizeof(int),1,physint);
    outphyint=state[5];
    printf("a1b1 %d\n", outphyint);
    fwrite(&outphyint,sizeof(int),1,physint);
    outphyint=state[6];
    printf("b1a1 %d\n", outphyint);
    fwrite(&outphyint,sizeof(int),1,physint);
    outphyint=state[7]+state[8]+state[9];
    printf("rd %d\n", outphyint);
    fwrite(&outphyint,sizeof(int),1,physint);
    outphyint=state[11];
    printf("dis %d\n", outphyint);
    fwrite(&outphyint,sizeof(int),1,physint);


    fclose(totalphy);
    fclose(physint);
    fclose(physfloat);

    float dep_sum=0,dep_total=0;
    FILE *depofp = NULL;
    fname = document["fileForEnergy"].GetString();
    depofp = fopen(fname.c_str(), "r");
    if (depofp == NULL)
    {
        printf("The file 'deposit.txt' doesn't exist, will be created and initialized as 0 0 \n");
    }
    else
    {
        fscanf(depofp, "%f %f", &dep_sum,&dep_total);
        fclose(depofp);
    }

    dep_sum += deposit_e;
    dep_total+= total_e;
    depofp = fopen(fname.c_str(), "w");
    fprintf(depofp, "%f %f", dep_sum, dep_total);
    fclose(depofp);
}

void PhysicsList::readDataToGPU()
{
	gFloat* ieeCSTable =(gFloat*) malloc(sizeof(gFloat)*(INTERACTION_TYPES-1)*E_ENTRIES);
    gFloat* DACSTable = (gFloat*) malloc(sizeof(gFloat)*DACS_ENTRIES*2);
    gFloat* elastDCSTable = (gFloat*) malloc(sizeof(gFloat)*E_ENTRIES*ODDS);  
    gFloat* BindE_array = (gFloat*)malloc(sizeof(gFloat)*BINDING_ITEMS*2);

    // read table
    rd_pioncs();
    rd_dacs( DACSTable);
    rd_ioncs( BindE_array, ieeCSTable);  
    rd_excit( &BindE_array[5], &ieeCSTable[5*E_ENTRIES]);
	rd_elast( &ieeCSTable[10*E_ENTRIES], elastDCSTable);
    

    // cuda memory	
    CUDA_CALL(cudaMallocArray(&dev_DACSTable, &channelDesc, DACS_ENTRIES, 2,0));
    CUDA_CALL(cudaMallocArray(&dev_BindE_array, &channelDesc, BINDING_ITEMS*2, 1,0));
    CUDA_CALL(cudaMallocArray(&dev_ieeCSTable, &channelDesc, E_ENTRIES, (INTERACTION_TYPES-1),0));
    CUDA_CALL(cudaMallocArray(&dev_elastDCSTable, &channelDesc, E_ENTRIES, ODDS,0));

    // cuda memory
	
    CUDA_CALL(cudaMemcpy2DToArray(dev_DACSTable    , 0, 0, DACSTable, DACS_ENTRIES * sizeof(gFloat), DACS_ENTRIES * sizeof(gFloat), 2, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy2DToArray(dev_BindE_array  , 0, 0, BindE_array, BINDING_ITEMS*2 * sizeof(gFloat), BINDING_ITEMS*2 * sizeof(gFloat), 1, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy2DToArray(dev_ieeCSTable   , 0, 0, ieeCSTable, E_ENTRIES * sizeof(gFloat), E_ENTRIES * sizeof(gFloat), (INTERACTION_TYPES-1),cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy2DToArray(dev_elastDCSTable, 0, 0, elastDCSTable, E_ENTRIES * sizeof(gFloat), E_ENTRIES * sizeof(gFloat), ODDS, cudaMemcpyHostToDevice));
	
	printf("Reading data and loading successfully!\n");

	// resource description -> from cuda memory
	struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0]   = cudaAddressModeClamp;
    texDesc.addressMode[1]   = cudaAddressModeClamp;
    texDesc.readMode         = cudaReadModeElementType;
    texDesc.filterMode       = cudaFilterModeLinear;

    struct cudaResourceDesc resD_protonTable;
    memset(&resD_protonTable, 0, sizeof(resD_protonTable));
    resD_protonTable.resType = cudaResourceTypeArray;
    resD_protonTable.res.array.array = dev_protonTable; // array data comes from dev_protonTable
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
	memset(&resD_elastDCSTable, 0, sizeof(resD_elastDCSTable));
	resD_elastDCSTable.resType = cudaResourceTypeArray;
	resD_elastDCSTable.res.array.array = dev_elastDCSTable; // array data comes from dev_elastDCSTable   
    
	CUDA_CALL(cudaCreateTextureObject(&texObj_protonTable     , &resD_protonTable,   &texDesc, NULL));
    CUDA_CALL(cudaCreateTextureObject(&texObj_DACSTable    , &resD_DACSTable,   &texDesc, NULL));
    CUDA_CALL(cudaCreateTextureObject(&texObj_BindE_array  , &resD_BindE_array, &texDesc, NULL));
    CUDA_CALL(cudaCreateTextureObject(&texObj_ieeCSTable   , &resD_ieeCSTable,  &texDesc, NULL));
    CUDA_CALL(cudaCreateTextureObject(&texObj_elastDCSTable, &resD_elastDCSTable, &texDesc, NULL));

    free(DACSTable);
    free(BindE_array);
    free(elastDCSTable);
    free(ieeCSTable);
    
}
void PhysicsList::rd_pioncs()
{
	std::string fname = document["pCSData"].GetString();
	int start, end;
    FILE* fpProton = fopen(fname.c_str(),"rb");
    if(fpProton == NULL)
    {
        printf("No such FILE %s\n", fname.c_str());
        exit(1);
    }
    else
    {
    	printf("Reading %s\n", fname.c_str());
        start = ftell(fpProton);
        fseek(fpProton,0,SEEK_END);
        end = ftell(fpProton);
        int nRows = (end-start)/4/12;
        gFloat* protonTable = (gFloat*) malloc(sizeof(gFloat)*(end-start)/4);
        fseek(fpProton,0,SEEK_SET);
        fread(protonTable, sizeof(gFloat)*(end-start)/4, 1,fpProton);
        fclose(fpProton);

		CUDA_CALL(cudaMallocArray(&dev_protonTable, &channelDesc, nRows, 11,0));
		CUDA_CALL(cudaMemcpy2DToArray(dev_protonTable, 0, 0, &protonTable[nRows], nRows*4,  nRows*4, 11, cudaMemcpyHostToDevice));
		printf("load data %s to GPU ok\n", fname.c_str());

		free(protonTable);
    }
}

void PhysicsList::rd_dacs( gFloat *DACSTable)
{
	char buffer[100];
	int i;
	std::string fname = document["eDACSData"].GetString();
	FILE *dacsFp = fopen(fname.c_str(), "r");
	if (dacsFp == NULL)
	{
		printf("The file %s was not opened\n", fname.c_str());
		exit(1);
	}
	else 
		printf("Reading %s\n", fname.c_str());

	fgets(buffer, sizeof(buffer), dacsFp);

	for (i = 0; i < DACS_ENTRIES; i++)
	{
		fscanf(dacsFp, "%f", &DACSTable[i]);
		fscanf(dacsFp, "%f", &DACSTable[DACS_ENTRIES+i]);
	}
	fclose(dacsFp);	
}

void PhysicsList::rd_ioncs( gFloat *ionBindEnergy, gFloat *ioncsTable)
{
	char buffer[100];
	int i, j;
	gFloat dump;
	std::string fname = document["eIonCSData"].GetString();
	FILE *ioncsFp = fopen(fname.c_str(), "r");
	if (ioncsFp == NULL)
	{
		printf("The file %s was not opened\n",fname.c_str());
		exit(1);
	}
	else 
		printf("Reading %s\n", fname.c_str());

	fgets(buffer, sizeof(buffer), ioncsFp);
	for (i = 0; i < BINDING_ITEMS; i++)
	{
		fscanf(ioncsFp, "%f", &ionBindEnergy[i]);
	}
	fclose(ioncsFp);	

	ioncsFp = fopen(fname.c_str(), "r");
	fgets(buffer, sizeof(buffer), ioncsFp);
	fgets(buffer, sizeof(buffer), ioncsFp);
	fgets(buffer, sizeof(buffer), ioncsFp);
 	for (i = 0; i < E_ENTRIES; i++)
	{
		fscanf(ioncsFp, "%f", &dump);
		for (j = 0; j < BINDING_ITEMS; j++)
		{
			fscanf(ioncsFp, "%f", &ioncsTable[j*E_ENTRIES+i]);
		}
	}
	fclose(ioncsFp);	
}

void PhysicsList::rd_excit(gFloat *excitBindEnergy, gFloat *excitCSTable)
{
	char buffer[100];
	int i, j;
	std::string fname = document["eExcCSData"].GetString();
	FILE *excitCSfp = fopen(fname.c_str(), "r");
	if (excitCSfp == NULL)
	{
		printf("The file %s was not opened\n",fname.c_str());
		exit(1);
	}
	else 
		printf("Reading %s\n", fname.c_str());
	
	fgets(buffer, sizeof(buffer), excitCSfp);
	for (i = 0; i < BINDING_ITEMS; i++)
	{
		fscanf(excitCSfp, "%f", &excitBindEnergy[i]);
	}
	fclose(excitCSfp);	
	
	excitCSfp = fopen(fname.c_str(), "r");
	fgets(buffer, sizeof(buffer), excitCSfp);
	fgets(buffer, sizeof(buffer), excitCSfp);
	fgets(buffer, sizeof(buffer), excitCSfp);
	for (i = 0; i < E_ENTRIES; i++)
	{
		for (j = 0; j < BINDING_ITEMS; j++)
		{
			fscanf(excitCSfp, "%f", &excitCSTable[j*E_ENTRIES+ i]);
		}
	}
	fclose(excitCSfp);	
}

void PhysicsList::rd_elast( gFloat *elastCSTable, gFloat *elastDCSTable)
{
	int i, j;
	std::string fname = document["eElaCSData"].GetString();
	FILE *elastCSfp = fopen(fname.c_str(), "r");
	if (elastCSfp == NULL)
	{
		printf("The file %s was not opened\n",fname.c_str());
		exit(1);
	}
	else 
		printf("Reading %s\n", fname.c_str());
	
	for (i = 0; i < E_ENTRIES; i++)
	{
		fscanf(elastCSfp, "%f", &elastCSTable[i]);
	}
	fclose(elastCSfp);
	
	fname = document["eElaDCSData"].GetString();
	FILE * elastDCSfp = fopen(fname.c_str(), "r");
	if (elastDCSfp == NULL)
	{
		printf("The file %s was not opened\n",fname.c_str());
		exit(1);
	}

	for (i = 0; i < E_ENTRIES; i++)
	{
		for (j = 0; j < ODDS; j++)
		{
			fscanf(elastDCSfp, "%f", &elastDCSTable[j*E_ENTRIES+ i]);
		}
	}
	fclose(elastDCSfp);
}

