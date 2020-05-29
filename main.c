#include <stdio.h>
#include <stdlib.h>

int main()
{
	float doseincell = 0; //Required dose	
	float benchE= 8711400*doseincell*0.5;//Change dose to energy
	
	int initialize=1;
	if(initialize)
	{		
		system("cd ./phy_stage/ && chmod +x ./compile_cuMC && ./compile_cuMC");	
		system("cd ./prechem_stage/ && chmod +x ./compile_cuMC && ./compile_cuMC");
		system("cd ./chem_stage/ && chmod +x ./compile_cuMC && ./compile_cuMC");
	}

	system("rm ./phy_stage/output/*");
	system("rm ./chem_stage/Results/*");
	float dep_sum_before=0,dep_sum_after=0;
	FILE *depofp=NULL;
	while(1)
	{
		depofp = fopen("./phy_stage/output/deposit.txt", "r");
	    if (depofp == NULL)
		{
			printf("The file 'deposit.txt' doesn't exist\n");
			dep_sum_before = 0;
		}
		else
		{
			fscanf(depofp, "%f", &dep_sum_before);
			fclose(depofp);
		}
		
		system("cd ./phy_stage/ && ./microMC ../config.txt");
		
		depofp = fopen("./phy_stage/output/deposit.txt", "r");
	    if (depofp == NULL)
		{
			printf("The file 'deposit.txt' doesn't exist\n");
			exit(1);
		}
		else
		{
			fscanf(depofp, "%f", &dep_sum_after);
			fclose(depofp);
		}

		if(dep_sum_after==dep_sum_before) continue;//pay attention to this one
	
		system("cd ./prechem_stage/ && ./prechem 0");//change 0 to the GPU index you want

		if(dep_sum_after>benchE) 
		{
			system("cp ./phy_stage/output/* ./chem_stage/Results/");
			system("cd ./chem_stage/ && ./chem 0 10000 1");//change 0 to the GPU index you want
			//change 10000 ps to time of chemical stage you want
			break;
		}
		else
		{
			system("cd ./chem_stage/ && ./chem 0 10000 0");
			//change 0 to the GPU index you want
			//change 10000 ps to time of chemical stage you want
		}		
	}
	return 0;
}
