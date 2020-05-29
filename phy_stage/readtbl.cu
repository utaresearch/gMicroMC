#include <stdio.h>
#include <stdlib.h>
#include "microMC.h"

void rd_dacs(FILE *dacsFp, REAL **DACSTable)
{
	char buffer[100];
	int i;
	dacsFp = fopen("./tables/DACS.txt", "r");
	if (dacsFp == NULL)
	{
		printf("The file 'DACS.txt' was not opened\n");
		exit(1);
	}

	fgets(buffer, sizeof(buffer), dacsFp);

	for (i = 0; i < DACS_ENTRIES; i++)
	{
		fscanf(dacsFp, "%f", &DACSTable[0][i]);
		fscanf(dacsFp, "%f", &DACSTable[1][i]);
	}
	fclose(dacsFp);	
}

void rd_ioncs(FILE *ioncsFp, REAL *ionBindEnergy, REAL **ioncsTable)
{
	char buffer[100];
	int i, j;
	REAL dump;
	ioncsFp = fopen("./tables/ionizationCS.txt", "r");
	if (ioncsFp == NULL)
	{
		printf("The file 'ionizationCS.txt' was not opened\n");
		exit(1);
	}

	fgets(buffer, sizeof(buffer), ioncsFp);
	for (i = 0; i < BINDING_ITEMS; i++)
	{
		fscanf(ioncsFp, "%f", &ionBindEnergy[i]);
	}
	fclose(ioncsFp);	

	ioncsFp = fopen("./tables/ionizationCS.txt", "r");
	fgets(buffer, sizeof(buffer), ioncsFp);
	fgets(buffer, sizeof(buffer), ioncsFp);
	fgets(buffer, sizeof(buffer), ioncsFp);
 	for (i = 0; i < E_ENTRIES; i++)
	{
		fscanf(ioncsFp, "%f", &dump);
		for (j = 0; j < BINDING_ITEMS; j++)
		{
			fscanf(ioncsFp, "%f", &ioncsTable[j][i]);
		}
	}
	fclose(ioncsFp);	
}

void rd_elast(FILE *elastCSfp, FILE *elastDCSfp, REAL **elastCSTable, REAL **elastDCSTable)
{
	int i, j;
	elastCSfp = fopen("./tables/elasticCS.txt", "r");
	if (elastCSfp == NULL)
	{
		printf("The file 'elasticCS.txt' was not opened\n");
		exit(1);
	}

	for (i = 0; i < E_ENTRIES; i++)
	{
		fscanf(elastCSfp, "%f", &elastCSTable[0][i]);
	}
	fclose(elastCSfp);
	
	elastDCSfp = fopen("./tables/elasticDCS.txt", "r");
	if (elastDCSfp == NULL)
	{
		printf("The file 'elasticDCS.txt' was not opened\n");
		exit(1);
	}

	for (i = 0; i < E_ENTRIES; i++)
	{
		for (j = 0; j < ODDS; j++)
		{
			fscanf(elastDCSfp, "%f", &elastDCSTable[j][i]);
		}
	}
	fclose(elastDCSfp);
}

void rd_excit(FILE *excitCSfp, REAL *excitBindEnergy, REAL **excitCSTable)
{
	char buffer[100];
	int i, j;
	excitCSfp = fopen("./tables/excitationCS.txt", "r");
	if (excitCSfp == NULL)
	{
		printf("The file 'excitationCS.txt' was not opened\n");
		exit(1);
	}

	fgets(buffer, sizeof(buffer), excitCSfp);
	for (i = 0; i < BINDING_ITEMS; i++)
	{
		fscanf(excitCSfp, "%f", &excitBindEnergy[i]);
	}
	fclose(excitCSfp);	
	
	excitCSfp = fopen("./tables/excitationCS.txt", "r");
	fgets(buffer, sizeof(buffer), excitCSfp);
	fgets(buffer, sizeof(buffer), excitCSfp);
	fgets(buffer, sizeof(buffer), excitCSfp);
	for (i = 0; i < E_ENTRIES; i++)
	{
		for (j = 0; j < BINDING_ITEMS; j++)
		{
			fscanf(excitCSfp, "%f", &excitCSTable[j][i]);
		}
	}
	fclose(excitCSfp);	
}











