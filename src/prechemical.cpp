#include "prechemical.h"

PrechemList::PrechemList()
{
	readBranchInfo(document["fileForBranchInfo"].GetString());
	readThermRecombInfo(document["fileForRecombineInfo"].GetString());
	readWaterStates();
}

PrechemList::~PrechemList()
{
	free(pb_rece);
	free(pb_wi); 
	free(pb_we_a1b1);
	free(pb_we_b1a1);
	free(pb_we_rd);
	free(pb_w_dis);	
	free(btype_rece);
	free(btype_wi);
	free(btype_we_a1b1);
	free(btype_we_b1a1);
	free(btype_we_rd);
	free(btype_w_dis);	
	free(num_product_btype);
	free(ptype_product_btype);
	free(placeinfo_btype);

	free(p_recomb_elec);
	free(rms_therm_elec);

	free(posx);
	free(posy);
	free(posz);
	free(ene);
	free(ttime);
	free(statetype);
	free(index);
	free(wiid_elec);
}

void PrechemList::readBranchInfo(std::string fname)
{
	char buffer[256];
	FILE *fp = fopen(fname.c_str(), "r");   
    printf("\n\nloading %s\n", fname.c_str());
	
	fgets(buffer, 250, fp);	
	fscanf(fp, "%d\n", &nbtype);
	num_product_btype = (int*) malloc(sizeof(int) * nbtype);
	ptype_product_btype = (int*) malloc(sizeof(int) * nbtype * MAXNUMBRANCHPROD);	
	fgets(buffer, 250, fp);
	
	int temp, i, k;	
	for(i=0; i<nbtype; i++)
	{
	  fscanf(fp, "%d %d", &temp, &num_product_btype[i]);
	  for(k=0; k<num_product_btype[i]; k++)
	  {
	    fscanf(fp, "%d", &ptype_product_btype[i* MAXNUMBRANCHPROD + k]);
	  }
	}
	fscanf(fp, "\n");

	placeinfo_btype = (float*)malloc(sizeof(float) * nbtype * 9);

	fgets(buffer, 250, fp);
	fgets(buffer, 250, fp);
	
    for(i=0; i<nbtype; i++)
	{
	    fscanf(fp, "%d", &temp);
		for(k=0; k<9; k++)
		{
		    fscanf(fp, "%f", &placeinfo_btype[i*9+k]);
		}
	}
	
	// loading the branches info for recombined electrons
	fscanf(fp, "\n");
	fgets(buffer, 250, fp);
	fscanf(fp, "%d", &nb_rece);	
	pb_rece = (float*) malloc(sizeof(float) * nb_rece);
	btype_rece = (int*) malloc(sizeof(int) * nb_rece);	
	for(i=0; i<nb_rece; i++)
	{
	  fscanf(fp, "%d %f", &btype_rece[i], &pb_rece[i]);
	}

	// loading the branches info for ionized water molecule
	fscanf(fp, "\n");
	fgets(buffer, 250, fp);
	fscanf(fp, "%d", &nb_wi);	
	pb_wi = (float*) malloc(sizeof(float) * nb_wi);
	btype_wi = (int*) malloc(sizeof(int) * nb_wi);	
	for(i=0; i<nb_wi; i++)
	{
	  fscanf(fp, "%d %f", &btype_wi[i], &pb_wi[i]);
	}
	
	// loading the branches info for A1B1 excited water molecule
	fscanf(fp, "\n");
	fgets(buffer, 250, fp);
	fscanf(fp, "%d", &nb_we_a1b1);
	pb_we_a1b1 = (float*)malloc(sizeof(float) * nb_we_a1b1);
	btype_we_a1b1 = (int*)malloc(sizeof(int) * nb_we_a1b1);	
	for(i=0; i<nb_we_a1b1; i++)
	{
	  fscanf(fp, "%d %f", &btype_we_a1b1[i], &pb_we_a1b1[i]);
	}
	
	// loading the branches info for B1A1 excited water molecule
	fscanf(fp, "\n");
	fgets(buffer, 250, fp);
	fscanf(fp, "%d", &nb_we_b1a1);
	pb_we_b1a1 = (float*)malloc(sizeof(float) * nb_we_b1a1);
	btype_we_b1a1 = (int*)malloc(sizeof(int) * nb_we_b1a1);	
	for(i=0; i<nb_we_b1a1; i++)
	{
	  fscanf(fp, "%d %f", &btype_we_b1a1[i], &pb_we_b1a1[i]);
	}
	
	// loading the branches info for the excited water molecule with Rydberg and diffusion bands
	fscanf(fp, "\n");
	fgets(buffer, 250, fp);
	fscanf(fp, "%d", &nb_we_rd);	
	pb_we_rd = (float*)malloc(sizeof(float) * nb_we_rd);
	btype_we_rd = (int*)malloc(sizeof(int) * nb_we_rd);	
	for(i=0; i<nb_we_rd; i++)
	{
	  fscanf(fp, "%d %f", &btype_we_rd[i], &pb_we_rd[i]);
	}
	
	// loading the branches info for the dissociative water molecule 
	fscanf(fp, "\n");
	fgets(buffer, 250, fp);	
	fscanf(fp, "%d", &nb_w_dis);
	pb_w_dis = (float*)malloc(sizeof(float) * nb_w_dis);
	btype_w_dis = (int*)malloc(sizeof(int) * nb_w_dis);	
	for(i=0; i<nb_w_dis; i++)
	{
	  fscanf(fp, "%d %f", &btype_w_dis[i], &pb_w_dis[i]);
	}
	fclose(fp);

	if(verbose>1)
	{
		printf("Information is listed in the following\n");
		printf("number of branches\n%d\n",nbtype);
		for(i =0; i<nbtype;i++)
		{
			printf("type %d has %d products: ", i, num_product_btype[i]);
			for(k=0;k<num_product_btype[i];k++)
			{
				printf("%d ", ptype_product_btype[i*MAXNUMBRANCHPROD+k]);
			}
			printf("\n");
		}
		
		printf("Brach types information for recombined electrons\n");
		for(i =0; i<nb_rece;i++)
		{
			printf("    Branch %d prob %f\n", btype_rece[i], pb_rece[i]);
		}
	}
}

void PrechemList::readThermRecombInfo(std::string fname)
{
	char buffer[256];
	FILE *fp = fopen(fname.c_str(), "r");   
    printf("\n\nloading %s\n", fname.c_str());
	
	fgets(buffer, 250, fp);
	fscanf(fp, "%d\n", &nebin);	
	
	p_recomb_elec = (float*)malloc(sizeof(float) * nebin);
	rms_therm_elec = (float*)malloc(sizeof(float) * nebin);	
	float *temp = (float*)malloc(sizeof(float) * nebin); // for energy entries

	fgets(buffer, 250, fp);
	for(int i=0; i<nebin; i++)
	{
	    fscanf(fp, "%f %f %f\n", &temp[i], &p_recomb_elec[i], &rms_therm_elec[i]);
	}
	fclose(fp);

	mine_ebin = temp[0];
	ide_ebin = 1.0f/(temp[1]-temp[0]);	
	free(temp);

	if(verbose>1)
	{
		printf("Information is listed in the following\n");
		printf("There are %d entries for thermolizing electrons\n",nebin);
		/*for(int i=0;i<nebin;i++)
		{
			printf("energy %f prob %f rms %f nm\n",mine_ebin+i/ide_ebin,p_recomb_elec[i], rms_therm_elec[i]);
		}*/
	}
}

void PrechemList::readWaterStates()
{
	std::string fname = document["fileForIntInput"].GetString();
	FILE* fpint=fopen(fname.c_str(), "rb");
    fname = document["fileForFloatInput"].GetString();
	FILE* fpfloat=fopen(fname.c_str(), "rb");
	int start, stop;
	start = ftell(fpfloat);
	fseek (fpfloat, 0, SEEK_END);
	stop = ftell(fpfloat);
	fseek (fpfloat, 0, SEEK_SET);
	printf("start=%d, end=%d",start,stop);
	num_total = (stop-start)/4/5;

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
	
	printf("the total number of initial reactant is %d\n", num_total);

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
 		
	wiid_elec = (int*)malloc(sizeof(int) * num_elec); 		
	int *tag_wiid = (int*)malloc(sizeof(int)*num_wi);
	memset(tag_wiid, 0, num_wi);

	float penergy, parposX, parposY, parposZ, etime;
    int parID, ptype, stype;	
	
	int idx_elec = 0;
	int idx_wi = 0;
	int idx_we_a1b1 = 0;
	int idx_we_b1a1 = 0;
	int idx_we_rd = 0; 
	int idx_w_dis = 0;

	for(int i = 0; i < num_total; i++)
	{
        if (i%10000==0)printf("i = %d/%d\n", i, num_total);

	    parID=phyint[4*i+1];
		ptype=phyint[4*i+2];
		stype=phyint[4*i+3];
		penergy=phyfloat[5*i];
		parposX=phyfloat[5*i+1];
		parposY=phyfloat[5*i+2];
		parposZ=phyfloat[5*i+3];
		etime = phyfloat[5*i+4];
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
		  posx[sidx_elec + idx_elec] = parposX;
		  posy[sidx_elec + idx_elec] = parposY;
		  posz[sidx_elec + idx_elec] = parposZ;
		  ene[sidx_elec + idx_elec] = penergy;
		  statetype[sidx_elec + idx_elec] = stype;	
		  index[sidx_elec + idx_elec] = parID;
		  ttime[sidx_elec + idx_elec] = etime;
		  float temp, mintemp = 1000000.f;
		  int idx_wiid;
		  
		  for(int k = sidx_wi; k < sidx_wi + idx_wi; k++) // need to be changed to GPU
		  {
		    temp = sqrt((parposX - posx[k]) * (parposX - posx[k]) + (parposY - posy[k]) * (parposY - posy[k]) + (parposZ - posz[k]) * (parposZ - posz[k]));			  
			  	
            if(temp < mintemp && tag_wiid[k - sidx_wi] == 0)
            {     				
			    mintemp = temp;
				idx_wiid = k;
			}
		  }
		  
		  tag_wiid[idx_wiid - sidx_wi] = 1;

		  wiid_elec[idx_elec] = idx_wiid;
		  
		  idx_elec++;
		}		
	}

	printf("idx_elec = %d, idx_wi = %d, idx_we_a1b1 = %d, idx_we_b1a1 = %d, idx_we_rd = %d, idx_w_dis = %d\n", idx_elec, idx_wi, idx_we_a1b1, idx_we_b1a1, idx_we_rd, idx_w_dis);
	
	if(idx_elec != num_elec || idx_wi != num_wi || idx_we_a1b1 != num_we_a1b1 || idx_we_b1a1 != num_we_b1a1 || idx_we_rd != num_we_rd || idx_w_dis != num_w_dis)
	{
	    printf("error in the number of the initial particles for prechemical stage.\n");
		exit(1);
	}
	
	free(tag_wiid);

	free(phyint);
	free(phyfloat);
}


