#include "chemical.h"

ChemList::ChemList()
{
	h_deltaT_adap[0] = 0.1f; // time step length array
	h_deltaT_adap[1] = 1.0f;
	h_deltaT_adap[2] = 3.0f;
	h_deltaT_adap[3] = 10.0f;
	h_deltaT_adap[4] = 100.0f;


    for(int i=0;i<21;i++) // Temperature array for reaction coefficient
        reactTemCoef[i] = 1;

	recordInterval = document["saveInterval"].GetFloat();
	elapseRecord = 0;
	reactInterval = document["DNAReactTime"].GetFloat();
	elapseReact = 0;

	iniPar = 0; // number of radicals
	posx = NULL;
	posy = NULL;
	posz = NULL;
	ttime = NULL;
	ptype = NULL;
	index = NULL;

	readRadiolyticSpecies();
	
	readReactions();
	setReactantList();
	setReactRadius();
}

ChemList::~ChemList()
{
	delete[] diffCoef_spec;
	delete[] radii_spec;

	cudaFree(d_posx);
	cudaFree(d_posy);
    cudaFree(d_posz);
	cudaFree(d_ptype);
	cudaFree(d_ttime);
	cudaFree(d_index);
	
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
	cudaFree(d_nzBinidx);
	cudaFree(d_idxnzBin_neig);
	//cudaFree(d_numParBinidx);
	cudaFree(d_idxnzBin_numNeig);
	
	cudaFree(d_mintd_Par);
}

void ChemList::readRadiolyticSpecies()
{
	char buffer[256];
	std::string fname = document["fileForSpecInfo"].GetString();
	FILE *fp = fopen(fname.c_str(), "r");
	if (fp == NULL)
	{
		printf("The file %s was not opened\n", fname.c_str());
		exit(1);
	}
	
	fgets(buffer, 100, fp);
	fscanf(fp, "%d\n", &numSpecType);

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
	   diffCoef_spec[i] = diffCoef_spec[i]*diffTemCoef;
	   Name_spec.push_back(specName);	   
	   if (maxDiffCoef_spec < diffCoef_spec[i])
		   maxDiffCoef_spec = diffCoef_spec[i];		 
	}	
	fclose(fp);	

	if(verbose>1)
	{
		printf("File %s was read as the following\n",fname.c_str());
		printf("In total %d species\n",numSpecType);
		for(int i=0;i<numSpecType;i++)
		{
			printf("Species %s diffusion rate %f nm^2/ps nominal reaction radius %f nm\n",Name_spec[i].c_str(),diffCoef_spec[i],radii_spec[i]);
		}
	}
}


void ChemList::readReactions()
{
	char buffer[256];
	string fname = document["fileForReactionInfo"].GetString();
	FILE *fp = fopen(fname.c_str(), "r");
	if (fp == NULL)
	{
		printf("The file in line 116 was not opened\n");
		exit(1);
	}	
	fgets(buffer, 100, fp); 
	fscanf(fp, "%d\n", &numReact);

	fgets(buffer, 200, fp);

	int k1 = 0;
	int k2 = 0;
	int temp; 
	
	for (int i = 0; i < numReact; i++)
	{
		diffCoef_total_React[i] = 0.0f;		
		fscanf(fp, "%d ", &temp);				
		fscanf(fp, "%d ", &numReactant_React[i]);

		indexReactant_React[i] = k1;

		for (int j = 0; j < numReactant_React[i]; j++)
		{
			fscanf(fp, "%d ", &typeReactant_React[k1]);
			diffCoef_total_React[i] += diffCoef_spec[typeReactant_React[k1]];			
			k1++;
		}

		fscanf(fp, "%d ", &numNewPar_React[i]);

		indexNewPar_React[i] = k2;

		for (int j = 0; j < numNewPar_React[i]; j++)
		{
			fscanf(fp, "%d ", &typeNewPar_React[k2]);
			k2++;
		}

		fscanf(fp, "%e %f %f\n", &kobs_React[i], &radii_React[i], &prob_React[i]);
		kobs_React[i] = kobs_React[i]*reactTemCoef[i];		
	}

	indexReactant_React[numReact] = k1;
	indexNewPar_React[numReact] = k2;

	if(verbose>1)
    {
		printf("\nReaction list is reorganized as the following:\n");
        printf("Number of reactions list in the file: %d\n", numReact);    
        printf("Number of reactants for each reaction:\n");
        for (int i = 0; i < numReact; i++)
        {
            printf("%d ", numReactant_React[i]);
        }
        printf("\n");
        
        printf("type of reactants for each reaction:\n");
        for (int i = 0; i < numReact; i++)
        {
            for (int j = indexReactant_React[i]; j < indexReactant_React[i + 1]; j++)
            {
                printf("%d ", typeReactant_React[j]);
            }
            printf("\n");
        }

        printf("Number of new particles for each reaction:\n");
        for (int i = 0; i < numReact; i++)
        {
            printf("%d ", numNewPar_React[i]);
        }
        printf("\n");

        printf("type of new particles for each reaction:\n");
/*        for (int i = 0; i < numReact; i++)
        {
            for (int j = indexNewPar_React[i]; j < indexNewPar_React[i + 1]; j++)
            {
                printf("%d ", typeNewPar_React[j]);
            }
            printf("\n");
        }*/
    }
}
void ChemList::setReactantList()
//---------------------------------------------------------------------------------------
//reorgnize the reaction lists based on current particle index for searching the neighbors that can
//have a reaction with the current particle	
//----------------------------------------------------------------------------------------
{
	int i, j, k, idx;
	int tempParType;

	for (i = 0; i < numSpecType; i++)
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
					if (numReactant_Par[tempParType] == 0) // no reactants for current particle then create the list
					{
						typeReactant_Par[tempParType * MAXNUMREACTANT4PAR] = typeReactant_React[k];
						subtypeReact_Par[tempParType * MAXNUMREACTANT4PAR] = i;
						numReactant_Par[tempParType]++;
					}

					if (numReactant_Par[tempParType] > 0) // if there is already possible reactants for current particle, then add the new reactant
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

    if(verbose>1)
    {
        for (i = 0; i < numSpecType; i++)
        {
            printf("particle type: %d, number of potential reactants for this particle type: %d\n", i, numReactant_Par[i]);
            
            if (numReactant_Par[i] > 0)
            {
                printf("particle type of these potential reactants: \n");
                for (j = 0; j < numReactant_Par[i]; j++)
                {
                    printf("%d ", typeReactant_Par[i*MAXNUMREACTANT4PAR + j]);
                }
                printf("\n");

                printf("reaction type corresponds to these potential reactants: \n");
            /*    for (j = 0; j < numReactant_Par[i]; j++)
                {
                    printf("%d ", subtypeReact_Par[i*MAXNUMREACTANT4PAR + j]);
                }
                printf("\n");*/
            }
            printf("------------------------------------------------------------------------------------\n");
        }
    }
}

void ChemList::setReactRadius()
{
	int ireact, ideltaT;
	float radii;
	float temp;

	for(ireact = 0; ireact < numReact; ireact++)
	{
		radii = kobs_React[ireact]/756.8725/diffCoef_total_React[ireact];

		for(ideltaT = 0; ideltaT < NUMDIFFTIMESTEPS; ideltaT++)
		{		
			if(document["useConstantRadius"].GetInt() == 1)
			{
				calc_radii_React[ireact * NUMDIFFTIMESTEPS + ideltaT] = radii;
			}   
			else
			{
				temp = sqrt(PI * diffCoef_total_React[ireact] * h_deltaT_adap[ideltaT]);
				calc_radii_React[ireact * NUMDIFFTIMESTEPS + ideltaT] = sqrt(temp * temp * 0.25f + temp * radii) - temp * 0.5f;
			}
							
			if(max_calc_radii_React[ideltaT] < calc_radii_React[ireact * NUMDIFFTIMESTEPS + ideltaT])
				max_calc_radii_React[ideltaT] = calc_radii_React[ireact * NUMDIFFTIMESTEPS + ideltaT];
		}
	}
	if(verbose>1)
	{
		printf("\n\nIn total %d reactions\n",numReact);
		for(ireact=0; ireact < numReact; ireact++)
		{
			printf("Reaction %d radius ",ireact);
			for(ideltaT = 0; ideltaT < NUMDIFFTIMESTEPS; ideltaT++)
				printf("%f ",calc_radii_React[ireact * NUMDIFFTIMESTEPS + ideltaT]);
			printf("\n");
		}
	}
}

void ChemList::readIniRadicals()
{
	printf("inipar is %d test\n",iniPar);
	if(iniPar>0) {cleanRadicals();printf("clean success\n");}

	string fname = document["fileForRadicalInfo"].GetString();
	printf("Reading %s\n",fname.c_str());
	FILE *fp = fopen(fname.c_str(), "rb");
	fseek(fp,0,SEEK_END);
	iniPar = ftell(fp)/24;
	printf("inipar is %d test2\n",iniPar);
    posx = new float[iniPar];
    posy = new float[iniPar];
    posz = new float[iniPar];
    ptype = new unsigned char[iniPar];
    index= new int[iniPar];
    ttime = new float[iniPar];
	printf("inipar is %d test3\n",iniPar);
	fseek(fp,0,SEEK_SET);
	fread(posx, sizeof(float),iniPar, fp);
	fread(posy, sizeof(float),iniPar, fp);
	fread(posz, sizeof(float),iniPar, fp);
	fread(ttime, sizeof(float),iniPar, fp);
	fread(index, sizeof(int),iniPar, fp);
	for(int i=0;i<iniPar;i++)
		fread(&(ptype[i]), sizeof(int),1, fp);
	fclose(fp);
	
	if(verbose>0)
	{
		printf("Read initial radical information as the following\n");
		printf("Initial radical number is %d\n", iniPar);
		int ntype[numSpecType];
		for(int i=0;i<numSpecType;i++)
			ntype[i] = 0;
		for(int i=0;i<iniPar;i++)
		{
			ntype[ptype[i]]++;
		}
		for(int i=0;i<numSpecType;i++)
		{
			printf("type %d radcial/molecule number %d\n",i,ntype[i]);
		}
	}
}
void ChemList::cleanRadicals()
{
	iniPar = 0;
	delete[] posx;
	delete[] posy;
	delete[] posz;
	delete[] ptype;
	delete[] index;
	delete[] ttime;
	posx = NULL;
	posy = NULL;
	posz = NULL;
	ttime = NULL;
	ptype = NULL;
	index = NULL;
}


void ChemList::saveNvsTime()
{// for number of radicals vs time
    float*h_posx=(float*) malloc(sizeof(float) * numCurPar*2);
	float*h_posy=(float*) malloc(sizeof(float) * numCurPar*2);
	float*h_posz=(float*) malloc(sizeof(float) * numCurPar*2);	
	unsigned char*h_ptype=(unsigned char*) malloc(sizeof(unsigned char) * numCurPar*2);

	cudaMemcpy(h_posx,d_posx,sizeof(float)*numCurPar, cudaMemcpyDeviceToHost); 
	cudaMemcpy(h_posy,d_posy,sizeof(float)*numCurPar, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_posz,d_posz,sizeof(float)*numCurPar, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_ptype,d_ptype,sizeof(unsigned char)*numCurPar, cudaMemcpyDeviceToHost);
	
	float roi = document["chemROI"].GetFloat();
	int numOfSpec[2*numSpecType];
	for(int tmptmp =0;tmptmp<2*numSpecType;tmptmp++)
		numOfSpec[tmptmp] = 0;
	float r2=0;
	for(int tmptmp=0;tmptmp<numCurPar;tmptmp++)
	{
		if(h_ptype[tmptmp]>2*numSpecType) printf("\n\nSomething is wrong...\n\n");
		numOfSpec[h_ptype[tmptmp]]++;
		r2=h_posx[tmptmp]*h_posx[tmptmp]+h_posy[tmptmp]*h_posy[tmptmp]+h_posz[tmptmp]*h_posz[tmptmp];
		if(r2<roi*roi)
			numOfSpec[numSpecType+h_ptype[tmptmp]]++;
	}
	std::string fname = document["numberFileForNvsTime"].GetString();
	FILE* fpspecies=fopen(fname.c_str(),"ab");
	fwrite(&numOfSpec[0],sizeof(int),2*numSpecType,fpspecies);
	fclose(fpspecies);

	fname = document["timeFileForNvsTime"].GetString();
	fpspecies=fopen(fname.c_str(),"ab");
	fwrite(&curTime,sizeof(float),1,fpspecies);
	fclose(fpspecies);

	if(verbose>2)
	{
		printf("current time is %f\nradical number ",curTime);
		for(int i=0;i<numSpecType;i++)
			printf("%d ",numOfSpec[i]);
		printf("\n");
	}
}

void ChemList::saveResults()
{
	std::string fname = document["fileForChemOutput"].GetString();
	int totalIni=2*iniPar;
	float tmpchem = 1-document["probChem"].GetFloat();
	
	FILE* fpchem = fopen(fname.c_str(),"ab");
	printf("x= %f ,y= %f ,z= %f \n", recordposition[1].x,recordposition[1].y,recordposition[1].z);
	for(int ii=0;ii<totalIni;ii++)
	{
		//if(recordposition[ii].w>0)
		//{
			if (ii<10)
				printf("x= %f ,y= %f ,z= %f \n", recordposition[ii].x,recordposition[ii].y,recordposition[ii].z);
			fwrite (&recordposition[ii].x, sizeof(float), 1, fpchem );
			fwrite (&recordposition[ii].y, sizeof(float), 1, fpchem );
			fwrite (&recordposition[ii].z, sizeof(float), 1, fpchem );
			fwrite (&tmpchem, sizeof(float), 1, fpchem );
		//}
	}
	fclose(fpchem);
}