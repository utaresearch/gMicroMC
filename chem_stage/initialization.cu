#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
using namespace std;
#include <ctime>
#include <cuda.h>
#include <cuda_runtime.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "microMC_chem.h"

ChemistrySpec::ChemistrySpec()
{
}

ChemistrySpec::~ChemistrySpec()
{
}

void ChemistrySpec::initChemistry(string fileprefix)
{
	string fname;

	fname = fileprefix;
	fname += string("/RadiolyticSpecies.txt");
	readRadiolyticSpecies(fname);

}

void ChemistrySpec::readRadiolyticSpecies(string fname)
{
	char buffer[256];
	
	FILE *fp = fopen(fname.c_str(), "r");
	if (fp == NULL)
	{
		printf("The file in line 37 was not opened\n");
		exit(1);
	}
	
	fgets(buffer, 100, fp);
	fscanf(fp, "%d\n", &numSpecType);

	printf("%s\n", buffer);
	printf("%d\n", numSpecType);

	diffCoef_spec = new float[numSpecType];
	radii_spec = new float[numSpecType];

	fgets(buffer, 100, fp);

	char specName[50];
	int temp;

	maxDiffCoef_spec = 0;

	for (int i = 0; i < numSpecType; i++)
	{
       fscanf(fp, "%d %s %f %f\n", &temp, specName, &diffCoef_spec[i], &radii_spec[i]);

	   diffCoef_spec[i] = diffCoef_spec[i] * 1.0e-3f; // convert the diffusion coefficient from 10-9 m^2/s to nm^2/ps
	   
	   Name_spec.push_back(specName);
	   
	   if (maxDiffCoef_spec < diffCoef_spec[i])
		   maxDiffCoef_spec = diffCoef_spec[i];

		 
	  // printf("i = %d, name of the particle is %s, diffusion coefficient is %e\n", i, Name_spec[i].c_str(), diffCoef_spec[i]);
	}
	
	fclose(fp);
	
}

ReactionType::ReactionType()
{
    h_deltaT_adap[0] = 0.1f;
	h_deltaT_adap[1] = 1.0f;
	h_deltaT_adap[2] = 3.0f;
	h_deltaT_adap[3] = 10.0f;
	h_deltaT_adap[4] = 100.0f;

	max_calc_radii_React[0] = 0.f;
	max_calc_radii_React[1] = 0.f;
	max_calc_radii_React[2] = 0.f;
	max_calc_radii_React[3] = 0.f;
	max_calc_radii_React[4] = 0.f;
}

ReactionType::~ReactionType()
{
}

void ReactionType::initReaction(ChemistrySpec chemistrySpec, string fileprefix)
{
	string fname;
	fname = fileprefix;

	fname += string("_ReactionInfo_orig.txt");
	readReactionTypeInfo(chemistrySpec, fname);

	setReactantList_Par(chemistrySpec);
	
	calcReactionRadius(chemistrySpec);
}

void ReactionType::readReactionTypeInfo(ChemistrySpec chemistrySpec, string fname)
{
	char buffer[256];

	FILE *fp = fopen(fname.c_str(), "r");
	if (fp == NULL)
	{
		printf("The file in line 116 was not opened\n");
		exit(1);
	}
	
	fgets(buffer, 100, fp); 
	fscanf(fp, "%d\n", &numReact);
	
	printf("%s\n", buffer);
	printf("%d\n", numReact);

	fgets(buffer, 200, fp);


	int k1 = 0;
	int k2 = 0;

	int temp; 
	
	for (int i = 0; i < numReact; i++)
	{
		diffCoef_total_React[i] = 0.0f;
		
		fscanf(fp, "%d ", &temp);
		
		printf("i = %d, ", i);
		
		fscanf(fp, "%d ", &numReactant_React[i]);

		indexReactant_React[i] = k1;

		for (int j = 0; j < numReactant_React[i]; j++)
		{
			fscanf(fp, "%d ", &typeReactant_React[k1]);
			printf("%s ", chemistrySpec.Name_spec[typeReactant_React[k1]].c_str());
			if(j < numReactant_React[i]-1) printf("+");//*/
			diffCoef_total_React[i] += chemistrySpec.diffCoef_spec[typeReactant_React[k1]];
			
			k1++;
		}


		fscanf(fp, "%d ", &numNewPar_React[i]);

		indexNewPar_React[i] = k2;

		printf(" = ");
		
		for (int j = 0; j < numNewPar_React[i]; j++)
		{
			fscanf(fp, "%d ", &typeNewPar_React[k2]);
			
			if(typeNewPar_React[k2]!=255) 
				printf("%s ", chemistrySpec.Name_spec[typeNewPar_React[k2]].c_str());
			else
				printf("H2O ");
			if(j < numNewPar_React[i]-1) printf("+");//*/
			k2++;
		}

		fscanf(fp, "%e %f %f\n", &kobs_React[i], &radii_React[i], &prob_React[i]);
		printf("%e\n", kobs_React[i]);
		
		//printf("%e %f %f\n", kobs_React[i], radii_React[i], prob_React[i]);
		
	}

	indexReactant_React[numReact] = k1;
	indexNewPar_React[numReact] = k2;
	cout<<"total number of new particles: "<<indexNewPar_React[numReact]<<endl;
}

void ReactionType::setReactantList_Par(ChemistrySpec chemistrySpec)
//---------------------------------------------------------------------------------------
//reorgnize the data for each type of the particles for searching the neighbors that can
//have a reaction with the current particle	
//----------------------------------------------------------------------------------------
{
	int i, j, k, idx;
	int tempParType;

	for (i = 0; i < chemistrySpec.numSpecType; i++)
	{
		numReactant_Par[i] = 0;
	}
	
	for (i = 0; i < numReact; i++)
	{
		for (j = indexReactant_React[i]; j < indexReactant_React[i+1]; j++)
		{
			tempParType = typeReactant_React[j];		
			for (k = indexReactant_React[i]; k < indexReactant_React[i + 1]; k++)
			{
				if (k != j)
				{
					if (numReactant_Par[tempParType] == 0)
					{
						typeReactant_Par[tempParType * MAXNUMREACTANT4PAR] = typeReactant_React[k];
						subtypeReact_Par[tempParType * MAXNUMREACTANT4PAR] = i;
						numReactant_Par[tempParType]++;
					}

					if (numReactant_Par[tempParType] > 0)
					{
						for (idx = 0; idx < numReactant_Par[tempParType]; idx++)
						{
							if (typeReactant_React[k] == typeReactant_Par[tempParType * MAXNUMREACTANT4PAR + idx])
								break;
						}

						if (idx == numReactant_Par[tempParType])
						{
							typeReactant_Par[tempParType * MAXNUMREACTANT4PAR + numReactant_Par[tempParType]] = typeReactant_React[k];
							subtypeReact_Par[tempParType * MAXNUMREACTANT4PAR + numReactant_Par[tempParType]] = i;
							numReactant_Par[tempParType]++;
						}
					}

				}
			}
			
		}
	}
}

void ReactionType::calcReactionRadius(ChemistrySpec chemistrySpec)
{
    int ireact, ideltaT;
	float radii;
	float temp;
		
    for(ireact = 0; ireact < numReact; ireact++)
    {
	    radii = kobs_React[ireact]/1e10/756.8725/diffCoef_total_React[ireact];// radii_React[ireact];
		//printf("reaction %d, radii=%f\n", ireact, radii);
		
		temp = sqrt(PI * diffCoef_total_React[ireact]* h_deltaT_adap[NUMDIFFTIMESTEPS-1]);
		
		radii = radii*(1.0f + radii/temp);
		
        for(ideltaT = 0; ideltaT < NUMDIFFTIMESTEPS; ideltaT++)
		{		   
		   temp = sqrt(PI * diffCoef_total_React[ireact] * h_deltaT_adap[ideltaT]);
		   
		  // calc_radii_React[ireact * NUMDIFFTIMESTEPS + ideltaT] = radii;
		   
		   //calc_radii_React[ireact * NUMDIFFTIMESTEPS + ideltaT] = radii * 0.5f + sqrt(radii * radii * 0.25f + radii * temp);

		   calc_radii_React[ireact * NUMDIFFTIMESTEPS + ideltaT] = sqrt(temp * temp * 0.25f + temp * radii) - temp * 0.5f;
		      
		   if(max_calc_radii_React[ideltaT] < calc_radii_React[ireact * NUMDIFFTIMESTEPS + ideltaT])
		   	max_calc_radii_React[ideltaT] = calc_radii_React[ireact * NUMDIFFTIMESTEPS + ideltaT];
		   
		   //printf("ireact = %d, ideltaT = %d, calc_radii_React = %f\n", ireact, ideltaT, calc_radii_React[ireact * NUMDIFFTIMESTEPS + ideltaT]);
		}
		
    }
}

ParticleData::ParticleData()
{
  posx = new float[MAXNUMPAR];
  posy = new float[MAXNUMPAR];
  posz = new float[MAXNUMPAR];
  ptype = new unsigned char[MAXNUMPAR];

  index= new int[MAXNUMPAR];
  ttime = new float[MAXNUMPAR];
  
  converTable[0] = 0;
  converTable[1] = 1;
  converTable[2] = 2;
  converTable[3] = 3;
  converTable[4] = 4;
  converTable[5] = 5;
  converTable[6] = 6;
  converTable[7] = 7;
  converTable[8] = 8;
  converTable[9] = 9;
  
}

ParticleData::~ParticleData()
{
  delete[] posx;
  delete[] posy;
  delete[] posz;
  delete[] ptype;
  delete[] index;
  delete[] ttime;
}

void ParticleData::readInitialParticles_RITRACK(string fname)
{
	
	FILE *fp = fopen(fname.c_str(), "r");
	if (fp == NULL)
	{
		printf("The file in line 390 was not opened\n");
		exit(1);
	}
   
    printf("%s\n", fname.c_str());
	
    char buffer[256];
	
	fgets(buffer, 100, fp);
	printf("%s\n", buffer);
	
	fscanf(fp, "%d\n", &initnumPar);
	printf("%d\n", initnumPar);
	
	fgets(buffer, 100, fp);
	//printf("%s\n", buffer);
	
	initTime = 0.0f;
	float tempTime;
	int tempPtype;
	
	//FILE *fp1 = fopen("test.txt", "w");
	int k = 0; 
	float posx_temp, posy_temp, posz_temp;
	
	for (int i = 0; i < initnumPar; i++)
	{
        fscanf(fp, "%e %e %e %d %e", &posx_temp, &posy_temp, &posz_temp, &tempPtype, &tempTime);
       
	    //printf("%e %e %e %d %e\n", posx_temp, posy_temp, posz_temp, tempPtype, tempTime);
		
	    if(converTable[tempPtype-1]!= -1)
		{
	      ptype[k] = converTable[tempPtype-1];
	    
		  posx[k] = posx_temp * 0.1f; // Angstrom to nm
		  posy[k] = posy_temp * 0.1f;
		  posz[k] = posz_temp * 0.1f;
		
		  //fprintf(fp1, "%d %d %f %f %f\n", k, ptype[k], posx[k], posy[k], posz[k]);
		  //printf("%d %d %f %f %f %e\n", k, ptype[k], posx[k], posy[k], posz[k], tempTime);
		  		  
		   k++;
		}
		
		if(initTime < tempTime)
			initTime = tempTime;
	}
	
	initnumPar = k;
	printf("%d\n", initnumPar);
	
	printf("initTime = %e\n", initTime);
	
	fclose(fp);

}

void ParticleData::readInitialParticles_GEANT4(string fname) // load the results obtained from geant4-DNA
{
	FILE *fp = fopen(fname.c_str(), "rb");
	fseek(fp,0,SEEK_END);
	initnumPar = ftell(fp)/24;
	printf("Number of loaded particles is %d\n", initnumPar);
	fseek(fp,0,SEEK_SET);
	fread(posx, sizeof(float),initnumPar, fp);
	fread(posy, sizeof(float),initnumPar, fp);
	fread(posz, sizeof(float),initnumPar, fp);
	fread(ttime, sizeof(float),initnumPar, fp);
	fread(index, sizeof(int),initnumPar, fp);
	for(int i=0;i<initnumPar;i++)
		fread(&(ptype[i]), sizeof(int),1, fp);
	fclose(fp);
}

void initGPUVariables(ChemistrySpec *chemistrySpec, ReactionType *reactType, ParticleData *parData)
{
	
	//gpu variables from ChemistrySpec class
	printf("Start GPU memory initialization\n");
	CUDA_CALL(cudaMemcpyToSymbol(d_diffCoef_spec, chemistrySpec->diffCoef_spec, sizeof(float)*chemistrySpec->numSpecType, 0, cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpyToSymbol(d_radii_spec, chemistrySpec->radii_spec, sizeof(float)*chemistrySpec->numSpecType, 0, cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpyToSymbol(d_maxDiffCoef_spec, chemistrySpec->diffCoef_spec, sizeof(float), 0, cudaMemcpyHostToDevice));

	//gpu variables from ReactionType class
	CUDA_CALL(cudaMemcpyToSymbol(d_numReactant_React, reactType->numReactant_React, sizeof(int)*reactType->numReact, 0, cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpyToSymbol(d_indexReactant_React, reactType->indexReactant_React, sizeof(int)*(reactType->numReact + 1), 0, cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpyToSymbol(d_typeReactant_React, reactType->typeReactant_React, sizeof(int)*reactType->indexReactant_React[reactType->numReact], 0, cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpyToSymbol(d_numNewPar_React, reactType->numNewPar_React, sizeof(int)*reactType->numReact, 0, cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpyToSymbol(d_indexNewPar_React, reactType->indexNewPar_React, sizeof(int)*(reactType->numReact + 1), 0, cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpyToSymbol(d_typeNewPar_React, reactType->typeNewPar_React, sizeof(int)*reactType->indexNewPar_React[reactType->numReact], 0, cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpyToSymbol(d_numReactant_Par, reactType->numReactant_Par, sizeof(float)*chemistrySpec->numSpecType, 0, cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpyToSymbol(d_typeReactant_Par, reactType->typeReactant_Par, sizeof(float)*chemistrySpec->numSpecType*MAXNUMREACTANT4PAR, 0, cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpyToSymbol(d_subtypeReact_Par, reactType->subtypeReact_Par, sizeof(float)*chemistrySpec->numSpecType*MAXNUMREACTANT4PAR, 0, cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpyToSymbol(d_kobs_React, reactType->kobs_React, sizeof(float)*reactType->numReact, 0, cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpyToSymbol(d_calc_radii_React, reactType->calc_radii_React, sizeof(float)*reactType->numReact * NUMDIFFTIMESTEPS, 0, cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpyToSymbol(d_prob_React, reactType->prob_React, sizeof(float)*reactType->numReact, 0, cudaMemcpyHostToDevice));

	//gpu class from ParticleData class
	numCurPar = parData->initnumPar;
	iniPar = int(numCurPar * 2.1);

	CUDA_CALL(cudaMalloc((void **) &d_posx, sizeof(float)* iniPar));
	CUDA_CALL(cudaMemcpy(d_posx, parData->posx, sizeof(float)*numCurPar, cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMalloc((void **) &d_posy, sizeof(float)*iniPar));
	CUDA_CALL(cudaMemcpy(d_posy, parData->posy, sizeof(float)*numCurPar, cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMalloc((void **) &d_posz, sizeof(float)*iniPar));
	CUDA_CALL(cudaMemcpy(d_posz, parData->posz, sizeof(float)*numCurPar, cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMalloc((void **) &d_ptype, sizeof(unsigned char)*iniPar));
	CUDA_CALL(cudaMemset(d_ptype, 255, sizeof(unsigned char) * iniPar));
	CUDA_CALL(cudaMemcpy(d_ptype, parData->ptype, sizeof(unsigned char)*numCurPar, cudaMemcpyHostToDevice));
	
	CUDA_CALL(cudaMalloc((void **) &d_index, sizeof(int)*iniPar));
	CUDA_CALL(cudaMemcpy(d_index, parData->index, sizeof(int)*numCurPar, cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(d_index+numCurPar, parData->index, sizeof(int)*numCurPar, cudaMemcpyHostToDevice));

	CUDA_CALL(cudaMalloc((void **) &d_ttime, sizeof(float)*iniPar));
	CUDA_CALL(cudaMemcpy(d_ttime, parData->ttime, sizeof(float)*numCurPar, cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(d_ttime+numCurPar, parData->ttime, sizeof(float)*numCurPar, cudaMemcpyHostToDevice));

	CUDA_CALL(cudaMalloc((void **) &d_posx_s, sizeof(float)*int(numCurPar * 1.5)));
	CUDA_CALL(cudaMalloc((void **) &d_posy_s, sizeof(float)*int(numCurPar * 1.5)));
	CUDA_CALL(cudaMalloc((void **) &d_posz_s, sizeof(float)*int(numCurPar * 1.5)));
	CUDA_CALL(cudaMalloc((void **) &d_ptype_s, sizeof(unsigned int)*int(numCurPar * 1.5)));
	
	CUDA_CALL(cudaMalloc((void **) &d_posx_d, sizeof(float)*int(numCurPar * 1.5)));
	CUDA_CALL(cudaMalloc((void **) &d_posy_d, sizeof(float)*int(numCurPar * 1.5)));
	CUDA_CALL(cudaMalloc((void **) &d_posz_d, sizeof(float)*int(numCurPar * 1.5)));
	
	CUDA_CALL(cudaMalloc((void **) &d_posx_sd, sizeof(float)*int(numCurPar * 1.5)));
	CUDA_CALL(cudaMalloc((void **) &d_posy_sd, sizeof(float)*int(numCurPar * 1.5)));
	CUDA_CALL(cudaMalloc((void **) &d_posz_sd, sizeof(float)*int(numCurPar * 1.5)));
	
	CUDA_CALL(cudaMalloc((void **) &d_gridParticleHash, sizeof(unsigned long)*int(numCurPar * 1.5))); 
	CUDA_CALL(cudaMalloc((void **) &d_gridParticleIndex, sizeof(int)*int(numCurPar * 1.5)));
	CUDA_CALL(cudaMalloc((void **) &d_accumParidxBin, sizeof(int)* (MAXNUMNZBIN + 1)));
	CUDA_CALL(cudaMalloc((void **) &d_nzBinidx, sizeof(unsigned long)* MAXNUMNZBIN));
	
	
	CUDA_CALL(cudaMalloc((void **) &d_idxnzBin_neig, sizeof(int)* MAXNUMNZBIN * 27));
    CUDA_CALL(cudaMalloc((void **) &d_idxnzBin_numNeig, sizeof(int)* MAXNUMNZBIN));
    
	
	CUDA_CALL(cudaMalloc((void **) &d_mintd_Par, sizeof(float)*int(numCurPar * 1.5)));
	
	h_mintd_Par_init = new float[int(numCurPar * 1.5)];
	for(int i = 0; i< int(numCurPar * 1.5); i++)
	{
	   h_mintd_Par_init[i] = 1.0e6f;
	}
	
	int tempNumNewPar = 0;
	CUDA_CALL(cudaMemcpyToSymbol(d_numNewPar, &tempNumNewPar, sizeof(int), 0, cudaMemcpyHostToDevice));
	
	CUDA_CALL(cudaMalloc((void **) &d_statusPar, sizeof(unsigned char)*iniPar));

	iniCurPar = numCurPar;
	float aa=clock();
	inirngG(0);
	float bb=clock();
	printf("initialization of rand is %f\n", (bb-aa)/CLOCKS_PER_SEC);
	printf("Finish initialization of random number\n");
}
