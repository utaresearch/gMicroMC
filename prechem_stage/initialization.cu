/* 
/* 6/19 : removefscanf(fp, " %e %e %e", &curposX, &curposY, &curposZ);
*/

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

#include "microMC_prechem_global.h"


ParticleData_prechem::ParticleData_prechem()
{
  
}

ParticleData_prechem::~ParticleData_prechem()
{

   free(posx);
   free(posy);
   free(posz);
   free(ene);
   free(ttime);
   free(statetype);
   free(index);
   //free(parposx_elec);
   //free(parposy_elec);
   //free(parposz_elec);
   free(wiid_elec);
 }

void ParticleData_prechem::readInitialParticles_GEANT4(string fname) // load the initial particles obtained from geant4-DNA for prechemical stage simulation
{
/*	
	FILE *fp = fopen(fname.c_str(), "r");
   
    printf("\n\nloading %s\n", fname.c_str());
	
    char buffer[256];
	
	fgets(buffer, 250, fp);
	//printf("%s\n", buffer);
	fscanf(fp, "%d\n", &num_elec);
	//printf("%d\n", num_elec);
	
	fgets(buffer, 250, fp);
	//printf("%s\n", buffer);
	fscanf(fp, "%d\n", &num_wi);
	//printf("%d\n", num_wi);
	
	fgets(buffer, 250, fp);
	//printf("%s\n", buffer);
	fscanf(fp, "%d\n", &num_we_a1b1);
	//printf("%d\n", num_we_a1b1);
	
	fgets(buffer, 250, fp);
	//printf("%s\n", buffer);
	fscanf(fp, "%d\n", &num_we_b1a1);
	//printf("%d\n", num_we_b1a1);
	
	fgets(buffer, 250, fp);
	//printf("%s\n", buffer);
	fscanf(fp, "%d\n", &num_we_rd);
	//printf("%d\n", num_we_rd);
	
	fgets(buffer, 250, fp);
	//printf("%s\n", buffer);
	fscanf(fp, "%d\n", &num_w_dis);
	//printf("%d\n", num_w_dis);
*/
	FILE* fpint=fopen("../phy_stage/output/phyint.dat", "rb");
	FILE* fpfloat=fopen("../phy_stage/output/phyfloat.dat", "rb");
	int start, stop;
	start = ftell(fpfloat);
	fseek (fpfloat, 0, SEEK_END);
	stop = ftell(fpfloat);
	fseek (fpfloat, 0, SEEK_SET);
	
	num_total = (stop-start)/4/5;
	printf("the total number of initial reactant is %d\n", num_total);

	int* phyint = (int*)malloc(sizeof(int) * num_total*4);
	float* phyfloat = (float*)malloc(sizeof(float) * num_total*5);
	if(phyfloat && phyint)
	{
		fread(phyint,sizeof(int),4*num_total,fpint);
		fread(phyfloat,sizeof(float),5*num_total,fpfloat);
	}
	else
	{
		printf("Wrong input!!!\n");
		exit(1);
	}
	fread(&num_elec,sizeof(int),1,fpint);
	fread(&num_wi,sizeof(int),1,fpint);
	fread(&num_we_a1b1,sizeof(int),1,fpint);
	fread(&num_we_b1a1,sizeof(int),1,fpint);
	fread(&num_we_rd,sizeof(int),1,fpint);
	fread(&num_w_dis,sizeof(int),1,fpint);

	fclose(fpint);
	fclose(fpfloat);
	
	sidx_elec = 0; // starting index for the solvated electrons in the particle array
	sidx_wi = num_elec;
	sidx_we_a1b1 = sidx_wi + num_wi; 
	sidx_we_b1a1 = sidx_we_a1b1 + num_we_a1b1; 
	sidx_we_rd = sidx_we_b1a1 + num_we_b1a1; 
	sidx_w_dis = sidx_we_rd + num_we_rd; 
	
	index = (int*) malloc(sizeof(int) * num_total);
	statetype = (int*)malloc(sizeof(int) * num_total);
	posx = (float*)malloc(sizeof(float) * num_total);
	posy = (float*)malloc(sizeof(float) * num_total);
	posz = (float*)malloc(sizeof(float) * num_total);
	ene = (float*)malloc(sizeof(float) * num_total);
	ttime = (float*)malloc(sizeof(float) * num_total);
	
	
	//parposx_elec = (float*)malloc(sizeof(float) * num_elec); 
	//parposy_elec = (float*)malloc(sizeof(float) * num_elec); 
	//parposz_elec = (float*)malloc(sizeof(float) * num_elec); 		
	
	wiid_elec = (int*)malloc(sizeof(int) * num_elec); 		
/*	
	fgets(buffer, 250, fp);
	//printf("%s\n", buffer);
	fgets(buffer, 250, fp);
	//printf("%s\n", buffer);
	fgets(buffer, 250, fp);
	//printf("%s\n", buffer);
	fgets(buffer, 250, fp);
	//printf("%s\n", buffer);
	fgets(buffer, 250, fp);
	//printf("%s\n", buffer);
//*/	
	
	float penergy, parposX, parposY, parposZ, etime;//curposX, curposY, curposZ, 
    int parID, ptype, stype;	
	
	int idx_elec = 0;
	int idx_wi = 0;
	int idx_we_a1b1 = 0;
	int idx_we_b1a1 = 0;
	int idx_we_rd = 0; 
	int idx_w_dis = 0;
	
	int *tag_wiid = (int*)malloc(sizeof(int)*num_wi);
	memset(tag_wiid, 0, num_wi);
	
/*	
	physicQ *loadQ = (physicQ*)malloc(sizeof(physicQ)*num_total);	

	for(int i = 0; i < num_total; i++)
	{
	    fscanf(fp, "%d %d %d %f %e %e %e %e", &loadQ[i].parID, &loadQ[i].ptype, &loadQ[i].stype, &loadQ[i].penergy, &loadQ[i].parposX, &loadQ[i].parposY, &loadQ[i].parposZ, &loadQ[i].etime);
    }		
    fclose(fp);
*/	
	
	
	for(int i = 0; i < num_total; i++)
	{
        if (i%10000==0)printf("i = %d/%d\n", i, num_total);
//	    fscanf(fp, "%d %d %d %f %e %e %e", &parID, &ptype, &stype, &penergy, &parposX, &parposY, &parposZ);
//	    fscanf(fp, "%d %d %d %f %e %e %e %e", &parID, &ptype, &stype, &penergy, &parposX, &parposY, &parposZ, &etime);
	    parID=phyint[4*i+1];// loadQ[i].parID;
		ptype=phyint[4*i+2];//loadQ[i].ptype;
		stype=phyint[4*i+3];//loadQ[i].stype;
		penergy=phyfloat[5*i];//loadQ[i].penergy;
		parposX=phyfloat[5*i+1];//loadQ[i].parposX;
		parposY=phyfloat[5*i+2];//loadQ[i].parposY;
		parposZ=phyfloat[5*i+3];//loadQ[i].parposZ;
		etime = phyfloat[5*i+4];//time
		//if(i<10)
		//	printf("particle state is %d %d %d %e %e %e %e %e\n", parID, ptype, stype, penergy, parposX, parposY, parposZ,etime);
        
		if(ptype == 7) //water molecule
		{
		  if(stype <= 4)
		  {
		    posx[sidx_wi + idx_wi] = parposX;
			posy[sidx_wi + idx_wi] = parposY;
			posz[sidx_wi + idx_wi] = parposZ;
			ene[sidx_wi + idx_wi] = penergy;
			statetype[sidx_wi + idx_wi] = stype;
			index[sidx_wi + idx_wi] = parID;
			ttime[sidx_wi + idx_wi] = etime;
			idx_wi++;
		  }
		  else if(stype == 5)
		  {
		    posx[sidx_we_a1b1 + idx_we_a1b1] = parposX;
			posy[sidx_we_a1b1 + idx_we_a1b1] = parposY;
			posz[sidx_we_a1b1 + idx_we_a1b1] = parposZ;
			ene[sidx_we_a1b1 + idx_we_a1b1] = penergy;
			statetype[sidx_we_a1b1 + idx_we_a1b1] = stype;
			index[sidx_we_a1b1 + idx_we_a1b1] = parID;
			ttime[sidx_we_a1b1 + idx_we_a1b1] = etime;
			idx_we_a1b1++;
		  }
		   else if(stype == 6)
		  {
		    posx[sidx_we_b1a1 + idx_we_b1a1] = parposX;
			posy[sidx_we_b1a1 + idx_we_b1a1] = parposY;
			posz[sidx_we_b1a1 + idx_we_b1a1] = parposZ;
			ene[sidx_we_b1a1 + idx_we_b1a1] = penergy;
			statetype[sidx_we_b1a1 + idx_we_b1a1] = stype;
			index[sidx_we_b1a1 + idx_we_b1a1] = parID;
			ttime[sidx_we_b1a1 + idx_we_b1a1] = etime;
			idx_we_b1a1++;
		  }
		   else if(stype >= 7 && stype <= 9)
		  {
		    posx[sidx_we_rd + idx_we_rd] = parposX;
			posy[sidx_we_rd + idx_we_rd] = parposY;
			posz[sidx_we_rd + idx_we_rd] = parposZ;
			ene[sidx_we_rd + idx_we_rd] = penergy;
			statetype[sidx_we_rd + idx_we_rd] = stype;
			//printf("idx_we_rd = %d  penergy = %e\n", idx_we_rd, penergy);
			index[sidx_we_rd + idx_we_rd] = parID;
			ttime[sidx_we_rd + idx_we_rd] = etime;
			idx_we_rd++;
		  }
		   else if(stype == 10)
		  {
		    posx[sidx_w_dis + idx_w_dis] = parposX;
			posy[sidx_w_dis + idx_w_dis] = parposY;
			posz[sidx_w_dis + idx_w_dis] = parposZ;
			ene[sidx_w_dis + idx_w_dis] = penergy;
			statetype[sidx_w_dis + idx_w_dis] = stype;
			index[sidx_w_dis + idx_w_dis] = parID;
			ttime[sidx_w_dis + idx_w_dis] = etime;
			idx_w_dis++;
		  }	  
		} 
		else if(ptype == 0) //solvated electron
		{
		 // fscanf(fp, " %e %e %e", &curposX, &curposY, &curposZ);
		 // printf(" %e %e %e", curposX, curposY, curposZ);
		  
		  posx[sidx_elec + idx_elec] = parposX;
		  posy[sidx_elec + idx_elec] = parposY;
		  posz[sidx_elec + idx_elec] = parposZ;
		  ene[sidx_elec + idx_elec] = penergy;
		  statetype[sidx_elec + idx_elec] = stype;	
		  index[sidx_elec + idx_elec] = parID;
		  ttime[sidx_elec + idx_elec] = etime;
		  float temp, mintemp = 1000000.f;
		  int idx_wiid;
		  
		  for(int k = sidx_wi; k < sidx_wi + idx_wi; k++)
		  {
		    temp = sqrt((parposX - posx[k]) * (parposX - posx[k]) + (parposY - posy[k]) * (parposY - posy[k]) + (parposZ - posz[k]) * (parposZ - posz[k]));			  
			  	
            if(temp < mintemp && tag_wiid[k - sidx_wi] == 0)
            {     				
			    mintemp = temp;
				idx_wiid = k;
			}
		  }
		  
		  tag_wiid[idx_wiid - sidx_wi] = 1;
		  
		 // printf("\n idx_wiid = %d, electron energy = %f, water energy = %f, water position = %e, %e, %e", idx_wiid, penergy, ene[idx_wiid], posx[idx_wiid], posy[idx_wiid], posz[idx_wiid]);
		  
		  //parposx_elec[idx_elec] = parposX;
		  //parposy_elec[idx_elec] = parposY;
		  //parposz_elec[idx_elec] = parposZ;
		  wiid_elec[idx_elec] = idx_wiid;
		  
		  idx_elec++;
		}		
		//printf("\n");
	}

	printf("idx_elec = %d, idx_wi = %d, idx_we_a1b1 = %d, idx_we_b1a1 = %d, idx_we_rd = %d, idx_w_dis = %d\n", idx_elec, idx_wi, idx_we_a1b1, idx_we_b1a1, idx_we_rd, idx_w_dis);
	
	if(idx_elec != num_elec || idx_wi != num_wi || idx_we_a1b1 != num_we_a1b1 || idx_we_b1a1 != num_we_b1a1 || idx_we_rd != num_we_rd || idx_w_dis != num_w_dis)
	{
	    printf("error in the number of the initial particles for prechemical stage.\n");
		exit(1);
	}
	
//	fclose(fp);
	
	free(tag_wiid);

	free(phyint);
	free(phyfloat);
/*	
	//debug
	fp = fopen("initial.bin", "wb");
	fwrite(posx, sizeof(float), num_total, fp);
    fwrite(posy, sizeof(float), num_total, fp);
	fwrite(posz, sizeof(float), num_total, fp);
	fwrite(statetype, sizeof(int), num_total, fp);
	fwrite(wiid_elec, sizeof(int), num_elec,fp);
	fclose(fp);//*/
}

Branch_water_prechem::Branch_water_prechem()
{
  
}

Branch_water_prechem::~Branch_water_prechem()
{

	free(pb_wi); 
	free(pb_we_a1b1);
	free(pb_we_b1a1);
	free(pb_we_rd);
	free(pb_w_dis);
	
	free(btype_wi);
	free(btype_we_a1b1);
	free(btype_we_b1a1);
	free(btype_we_rd);
	free(btype_w_dis);
	
	free(num_product_btype);
	free(ptype_product_btype);
	free(placeinfo_btype);
}

void Branch_water_prechem::readBranchInfo(string fname)
{
    FILE *fp = fopen(fname.c_str(), "r");
   
    printf("\n\nloading %s\n", fname.c_str());
	
    char buffer[256];
	
	//loading the info for all the branch types
	fgets(buffer, 250, fp);
	//printf("%s\n", buffer);
	fscanf(fp, "%d\n", &nbtype);
	//printf("%d\n", nbtype);
	
	num_product_btype = (int*)malloc(sizeof(int) * nbtype);
	ptype_product_btype = (int*)malloc(sizeof(int) * nbtype * MAXNUMBRANCHPROD);
	
	fgets(buffer, 250, fp);
	printf("%s\n", buffer);
	
	int temp, i, k;
	
	for(i=0; i<nbtype; i++)
	{
	  fscanf(fp, "%d %d", &temp, &num_product_btype[i]);
	  //printf("The %dth branch has %d products: ", temp, num_product_btype[i]);
	  for(k=0; k<num_product_btype[i]; k++)
	  {
	    fscanf(fp, "%d", &ptype_product_btype[i* MAXNUMBRANCHPROD + k]);
		//printf("%d ", ptype_product_btype[i* MAXNUMBRANCHPROD + k]);
	  }
	  //printf("\n");
	}
	
	placeinfo_btype = (float*)malloc(sizeof(float) * nbtype * 9);
	
	fscanf(fp, "\n");
	fgets(buffer, 250, fp);
	//printf("%s\n", buffer);
	fgets(buffer, 250, fp);
	//printf("%s\n", buffer);
	
    for(i=0; i<nbtype; i++)
	{
	    fscanf(fp, "%d", &temp);
		//printf("For the %dth branch: ", temp);
		for(k=0; k<9; k++)
		{
		    fscanf(fp, "%f", &placeinfo_btype[i*9+k]);
			//printf("%f ", placeinfo_btype[i*9+k]);
		}
		//printf("\n");
	}
	
	// loading the branches info for ionized water molecule
	fscanf(fp, "\n");
	fgets(buffer, 250, fp);
	//printf("%s\n", buffer);
	fscanf(fp, "%d", &nb_wi);
	//printf("%d ", nb_wi);
	
	pb_wi = (float*)malloc(sizeof(float) * nb_wi);
	btype_wi = (int*)malloc(sizeof(int) * nb_wi);
	
	for(i=0; i<nb_wi; i++)
	{
	  fscanf(fp, "%d %f", &btype_wi[i], &pb_wi[i]);
	  printf("%d %f ", btype_wi[i], pb_wi[i]);
	}
	printf("\n");
	
	// loading the branches info for A1B1 excited water molecule
	fscanf(fp, "\n");
	fgets(buffer, 250, fp);
	printf("%s\n", buffer);
	fscanf(fp, "%d", &nb_we_a1b1);
	printf("%d ", nb_we_a1b1);
	
	pb_we_a1b1 = (float*)malloc(sizeof(float) * nb_we_a1b1);
	btype_we_a1b1 = (int*)malloc(sizeof(int) * nb_we_a1b1);
	
	for(i=0; i<nb_we_a1b1; i++)
	{
	  fscanf(fp, "%d %f", &btype_we_a1b1[i], &pb_we_a1b1[i]);
	  //printf("%d %f ", btype_we_a1b1[i], pb_we_a1b1[i]);
	}
	printf("\n");
	
	// loading the branches info for B1A1 excited water molecule
	fscanf(fp, "\n");
	fgets(buffer, 250, fp);
	printf("%s\n", buffer);
	fscanf(fp, "%d", &nb_we_b1a1);
	printf("%d ", nb_we_b1a1);
	
	pb_we_b1a1 = (float*)malloc(sizeof(float) * nb_we_b1a1);
	btype_we_b1a1 = (int*)malloc(sizeof(int) * nb_we_b1a1);
	
	for(i=0; i<nb_we_b1a1; i++)
	{
	  fscanf(fp, "%d %f", &btype_we_b1a1[i], &pb_we_b1a1[i]);
	  //printf("%d %f ", btype_we_b1a1[i], pb_we_b1a1[i]);
	}
	printf("\n");
	
	// loading the branches info for the excited water molecule with Rydberg and diffusion bands
	fscanf(fp, "\n");
	fgets(buffer, 250, fp);
	printf("%s\n", buffer);
	fscanf(fp, "%d", &nb_we_rd);
	printf("%d ", nb_we_rd);
	
	pb_we_rd = (float*)malloc(sizeof(float) * nb_we_rd);
	btype_we_rd = (int*)malloc(sizeof(int) * nb_we_rd);
	
	for(i=0; i<nb_we_rd; i++)
	{
	  fscanf(fp, "%d %f", &btype_we_rd[i], &pb_we_rd[i]);
	  //printf("%d %f ", btype_we_rd[i], pb_we_rd[i]);
	}
	printf("\n");
	
	// loading the branches info for the dissociative water molecule 
	fscanf(fp, "\n");
	fgets(buffer, 250, fp);
	printf("%s\n", buffer);
	fscanf(fp, "%d", &nb_w_dis);
	printf("%d ", nb_w_dis);
	
	pb_w_dis = (float*)malloc(sizeof(float) * nb_w_dis);
	btype_w_dis = (int*)malloc(sizeof(int) * nb_w_dis);
	
	for(i=0; i<nb_w_dis; i++)
	{
	  fscanf(fp, "%d %f", &btype_w_dis[i], &pb_w_dis[i]);
	  //printf("%d %f ", btype_w_dis[i], pb_w_dis[i]);
	}
	printf("\n");
	
	fclose(fp);
	
}

ThermRecomb_elec_prechem::ThermRecomb_elec_prechem()
{
}

ThermRecomb_elec_prechem::~ThermRecomb_elec_prechem()
{
    free(p_recomb_elec);
	free(rms_therm_elec);
}

void ThermRecomb_elec_prechem::readThermRecombInfo(string fname)
{
    FILE *fp = fopen(fname.c_str(), "r");
   
    printf("\n\nloading %s\n", fname.c_str());
	
    char buffer[256];
	
	//loading the info for all the branch types
	fgets(buffer, 250, fp);
	printf("%s\n", buffer);
	fscanf(fp, "%d\n", &nebin);
	printf("%d\n", nebin);
	
	fgets(buffer, 250, fp);
	//printf("%s\n", buffer);
	
	p_recomb_elec = (float*)malloc(sizeof(float) * nebin);
	rms_therm_elec = (float*)malloc(sizeof(float) * nebin);
	
	float *temp = (float*)malloc(sizeof(float) * nebin);
	
	for(int i=0; i<nebin; i++)
	{
	    fscanf(fp, "%f %f %f\n", &temp[i], &p_recomb_elec[i], &rms_therm_elec[i]);
		
		//if(i%100 == 0)
		//printf("%f %f %f\n", temp[i], p_recomb_elec[i], rms_therm_elec[i]);
	}
	
	mine_ebin = temp[0];
	ide_ebin = 1.0f/(temp[1]-temp[0]);
	
	free(temp);
}


void initGPUVariables_pc(ParticleData_prechem *parData_pc, Branch_water_prechem *braInfo_pc, ThermRecomb_elec_prechem *thermRecombInfo_pc)
{
    float *temp_pos = (float*)malloc(sizeof(float) * parData_pc->num_total * 2);
	for(int i=0; i<parData_pc->num_total * 2; i++)
		temp_pos[i] = 10000;
	cudaMalloc((void **) &d_posx, sizeof(float)* parData_pc->num_total * 3); // one initial particle may produce up to 3 products, so triple memory size to include the products
	cudaMemcpy(d_posx, parData_pc->posx, sizeof(float)*parData_pc->num_total, cudaMemcpyHostToDevice);
	cudaMemcpy(d_posx+parData_pc->num_total, temp_pos, sizeof(float)*parData_pc->num_total*2, cudaMemcpyHostToDevice);
	
	cudaMalloc((void **) &d_posy, sizeof(float)* parData_pc->num_total * 3);
	cudaMemcpy(d_posy, parData_pc->posy, sizeof(float)*parData_pc->num_total, cudaMemcpyHostToDevice);
	cudaMemcpy(d_posy+parData_pc->num_total, temp_pos, sizeof(float)*parData_pc->num_total*2, cudaMemcpyHostToDevice);
	
	cudaMalloc((void **) &d_posz, sizeof(float)* parData_pc->num_total * 3);
	cudaMemcpy(d_posz, parData_pc->posz, sizeof(float)*parData_pc->num_total, cudaMemcpyHostToDevice);
	cudaMemcpy(d_posz+parData_pc->num_total, temp_pos, sizeof(float)*parData_pc->num_total*2, cudaMemcpyHostToDevice);
	free(temp_pos);
	
	cudaMalloc((void **) &d_index, sizeof(int)* parData_pc->num_total * 3);
	cudaMemcpy(d_index, parData_pc->index, sizeof(int)*parData_pc->num_total, cudaMemcpyHostToDevice);
	cudaMemcpy(d_index+parData_pc->num_total, parData_pc->index, sizeof(int)*parData_pc->num_total, cudaMemcpyHostToDevice);
	cudaMemcpy(d_index+parData_pc->num_total*2, parData_pc->index, sizeof(int)*parData_pc->num_total, cudaMemcpyHostToDevice);
	
	cudaMalloc((void **) &d_ttime, sizeof(float)* parData_pc->num_total * 3);
	cudaMemcpy(d_ttime, parData_pc->ttime, sizeof(float)*parData_pc->num_total, cudaMemcpyHostToDevice);
	cudaMemcpy(d_ttime+parData_pc->num_total, parData_pc->ttime, sizeof(float)*parData_pc->num_total, cudaMemcpyHostToDevice);
	cudaMemcpy(d_ttime+parData_pc->num_total*2, parData_pc->ttime, sizeof(float)*parData_pc->num_total, cudaMemcpyHostToDevice);

	int *temp_ptype = (int*)malloc(sizeof(int) * parData_pc->num_total * 3);
	for(int i=0; i<parData_pc->num_total * 3; i++)
		temp_ptype[i] = 255;	
	cudaMalloc((void **) &d_ptype, sizeof(int)* parData_pc->num_total * 3);
	cudaMemcpy(d_ptype, temp_ptype, sizeof(float)*parData_pc->num_total*3, cudaMemcpyHostToDevice);
	free(temp_ptype);
	
	cudaMalloc((void **) &d_statetype, sizeof(int)* parData_pc->num_total);
	cudaMemcpy(d_statetype, parData_pc->statetype, sizeof(int)*parData_pc->num_total, cudaMemcpyHostToDevice);
	
	cudaMalloc((void **) &d_ene, sizeof(float)* parData_pc->num_total); //since energy will not be considered at chemical stage, so only for the initial particles of the prechemical stage
	cudaMemcpy(d_ene, parData_pc->ene, sizeof(float)*parData_pc->num_total, cudaMemcpyHostToDevice);
	
	cudaMalloc((void **) &d_wiid_elec, sizeof(int)* parData_pc->num_elec); 
	cudaMemcpy(d_wiid_elec, parData_pc->wiid_elec, sizeof(int)*parData_pc->num_elec, cudaMemcpyHostToDevice);
	
    cudaMemcpyToSymbol(d_num_elec, &parData_pc->num_elec, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_num_wi, &parData_pc->num_wi, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_num_we_a1b1, &parData_pc->num_we_a1b1, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_num_we_b1a1, &parData_pc->num_we_b1a1, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_num_we_rd, &parData_pc->num_we_rd, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_num_w_dis, &parData_pc->num_w_dis, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_num_total, &parData_pc->num_total, sizeof(int), 0, cudaMemcpyHostToDevice);
	
	cudaMemcpyToSymbol(d_sidx_elec, &parData_pc->sidx_elec, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_sidx_wi, &parData_pc->sidx_wi, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_sidx_we_a1b1, &parData_pc->sidx_we_a1b1, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_sidx_we_b1a1, &parData_pc->sidx_we_b1a1, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_sidx_we_rd, &parData_pc->sidx_we_rd, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_sidx_w_dis, &parData_pc->sidx_w_dis, sizeof(int), 0, cudaMemcpyHostToDevice);

	cudaMemcpyToSymbol(d_nebin, &thermRecombInfo_pc->nebin, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_mine_ebin, &thermRecombInfo_pc->mine_ebin, sizeof(float), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_ide_ebin, &thermRecombInfo_pc->ide_ebin, sizeof(float), 0, cudaMemcpyHostToDevice);
	
	cudaMemcpyToSymbol(d_nbtype, &braInfo_pc->nbtype, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_placeinfo_btype, braInfo_pc->placeinfo_btype, sizeof(float)*braInfo_pc->nbtype*9, 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_num_product_btype, braInfo_pc->num_product_btype, sizeof(int)*braInfo_pc->nbtype, 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_ptype_product_btype, braInfo_pc->ptype_product_btype, sizeof(int)*braInfo_pc->nbtype * MAXNUMBRANCHPROD, 0, cudaMemcpyHostToDevice);
	
	cudaMemcpyToSymbol(d_nb_wi, &braInfo_pc->nb_wi, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_nb_we_a1b1, &braInfo_pc->nb_we_a1b1, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_nb_we_b1a1, &braInfo_pc->nb_we_b1a1, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_nb_we_rd, &braInfo_pc->nb_we_rd, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_nb_w_dis, &braInfo_pc->nb_w_dis, sizeof(int), 0, cudaMemcpyHostToDevice);
	
	cudaMemcpyToSymbol(d_pb_wi, braInfo_pc->pb_wi, sizeof(float)*braInfo_pc->nb_wi, 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_pb_we_a1b1, braInfo_pc->pb_we_a1b1, sizeof(float)*braInfo_pc->nb_we_a1b1, 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_pb_we_b1a1, braInfo_pc->pb_we_b1a1, sizeof(float)*braInfo_pc->nb_we_b1a1, 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_pb_we_rd, braInfo_pc->pb_we_rd, sizeof(float)*braInfo_pc->nb_we_rd, 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_pb_w_dis, braInfo_pc->pb_w_dis, sizeof(float)*braInfo_pc->nb_w_dis, 0, cudaMemcpyHostToDevice);
	
    cudaMemcpyToSymbol(d_btype_wi, braInfo_pc->btype_wi, sizeof(int)*braInfo_pc->nb_wi, 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_btype_we_a1b1, braInfo_pc->btype_we_a1b1, sizeof(int)*braInfo_pc->nb_we_a1b1, 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_btype_we_b1a1, braInfo_pc->btype_we_b1a1, sizeof(int)*braInfo_pc->nb_we_b1a1, 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_btype_we_rd, braInfo_pc->btype_we_rd, sizeof(int)*braInfo_pc->nb_we_rd, 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_btype_w_dis, braInfo_pc->btype_w_dis, sizeof(int)*braInfo_pc->nb_w_dis, 0, cudaMemcpyHostToDevice);
 
    cudaMallocArray(&d_p_recomb_elec, &p_recomb_elec_tex.channelDesc, thermRecombInfo_pc->nebin, 1);
    cudaMemcpyToArray(d_p_recomb_elec, 0, 0, thermRecombInfo_pc->p_recomb_elec, sizeof(float)*thermRecombInfo_pc->nebin, cudaMemcpyHostToDevice);
    p_recomb_elec_tex.filterMode = cudaFilterModeLinear;
    cudaBindTextureToArray(p_recomb_elec_tex, d_p_recomb_elec);
	
	cudaMallocArray(&d_rms_therm_elec, &rms_therm_elec_tex.channelDesc, thermRecombInfo_pc->nebin, 1);
    cudaMemcpyToArray(d_rms_therm_elec, 0, 0, thermRecombInfo_pc->rms_therm_elec, sizeof(float)*thermRecombInfo_pc->nebin, cudaMemcpyHostToDevice);
    rms_therm_elec_tex.filterMode = cudaFilterModeLinear;
    cudaBindTextureToArray(rms_therm_elec_tex, d_rms_therm_elec);
	
	inirngG(0);
	
}
