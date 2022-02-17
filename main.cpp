#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#include "global.h"
#include "initialize.h"
#include "physicsList.h"
#include "prechemical.h"
#include "chemical.h"
#include "DNAList.h"


int verbose, NPART, NRAND, NTHREAD_PER_BLOCK, deviceIndex;
Document document;
cudaDeviceProp devProp;

int main()
{
	system("mkdir -p ./output");
	system("rm ./output/*");
	std::string ss;
	std::stringstream buffer;
	std::ifstream fp("config.txt");	// With this line, just run ./gMicroMC, reads confi.txt everytime.	
	buffer << fp.rdbuf();
	fp.close();
	ss = buffer.str();
	std::cout<<ss<<std::endl;
	
	initialize(ss);

	float eneDeposited = 0;
	int irun = 0;
	PhysicsList pl;
	int nPar=0;
	nPar=document["nPar"].GetInt();
	int maxRun=0;
	maxRun=document["maxRun"].GetInt();
	while(irun<maxRun)//eneDeposited<document["targetEneDep"].GetFloat())
	{		
		// system("rm ./output/events.dat"); // uncomment this line if it givs file reading error
		pl.run();
		pl.saveResults();
		std::string fname = document["fileForEnergy"].GetString();
		float dep_sum=0,dep_total=0;
		FILE* depofp = fopen(fname.c_str(), "r");
	    if (depofp != NULL)
	    {
	        fscanf(depofp, "%f %f", &dep_sum, &dep_total);
	        fclose(depofp);
	    }
	    eneDeposited = dep_sum/1e6; //eneDeposited = dep_total; // depending on the region you want to use

		printf("total particle simulated is %d, total energy deposited in ROI is %f MeV, total energy deposited in world is %f MeV\n",(irun+1)*nPar, eneDeposited, dep_total/1e6);
		irun++;
		//if(irun>3) break; // uncomment this line when you are testing code for safety
	}

	PrechemList pcl;
	if(document["startStage"].GetInt()<2)
	{		
		pcl.initGPUVariables();
		pcl.run();
		pcl.saveResults();
	}

	
	ChemList cl;
	DNAList ddl;
	ddl.calDNAreact_radius(cl.diffCoef_spec);
	ddl.initDNA();

	if(document["startStage"].GetInt()<3)
	{	
		cl.readIniRadicals();
		cl.copyDataToGPU();	
		cl.run(ddl); // saveResutls function is called on the fly -- concurrent method 
	}

	
	/*if(document["startStage"].GetInt()<4)// || dose>targetDose)
	{	
		int repeat = document["repTimes"].GetInt();
		for(int jjj=0;jjj<repeat;jjj++)
		{	
			ddl.run();
			ddl.saveResults();
		}
	}*/
	CUDA_CALL(cudaFree(cuseed));
	return 0;
}