#include "DNAKernel.cuh"
__constant__  int neighborindex[27];
__constant__ float min1, min2, min3, max1, max2, max3;
__constant__  float d_rDNA[72];
void DNAList::initDNA()
{
	int totalspace = NUCLEUS_DIM*NUCLEUS_DIM*NUCLEUS_DIM;
	int *chromatinIndex = (int*)malloc(sizeof(int)*totalspace);
	int *chromatinStart = (int*)malloc(sizeof(int)*totalspace);
	int *chromatinType = (int*)malloc(sizeof(int)*totalspace);
	for (int k=0; k<totalspace; k++) 
	{
		chromatinIndex[k] = -1;
		chromatinStart[k] = -1;
		chromatinType[k] = -1;
	}

	int index, data[6];
	long lSize;
	printf("At least here!!\n");
	std::string fname = document["wholeDNA"].GetString();
	printf("Reading %s", fname.c_str());
	FILE* pFile=fopen(fname.c_str(),"rb");
	fseek (pFile , 0 , SEEK_END);
    lSize = ftell (pFile);
  	rewind (pFile);
  	for (int i=0; i<lSize/(6*sizeof(int)); i++)
	{
	    fread(data,sizeof(int),6, pFile);
	    //if(i<5) printf("%d %d %d %d %d %d\n", data[0], data[1], data[2], data[3], data[4], data[5]);
		index = data[0] + data[1] * NUCLEUS_DIM + data[2] * NUCLEUS_DIM * NUCLEUS_DIM;
		chromatinIndex[index] = data[3];
		chromatinStart[index] = data[4];
		chromatinType[index] = data[5];
		if(data[5]>30) printf("\n\n\n\nread in Wrong\n\n\n\n");
	}
	fclose(pFile);

	CUDA_CALL(cudaMalloc((void**)&dev_chromatinIndex, totalspace * sizeof(int)));
	CUDA_CALL(cudaMemcpy(dev_chromatinIndex, chromatinIndex, totalspace * sizeof(int), cudaMemcpyHostToDevice));//DNA index
	CUDA_CALL(cudaMalloc((void**)&dev_chromatinStart, totalspace * sizeof(int)));
	CUDA_CALL(cudaMemcpy(dev_chromatinStart, chromatinStart, totalspace * sizeof(int), cudaMemcpyHostToDevice));//# of start base in the box
	CUDA_CALL(cudaMalloc((void**)&dev_chromatinType, totalspace * sizeof(int)));
	CUDA_CALL(cudaMemcpy(dev_chromatinType, chromatinType, totalspace * sizeof(int), cudaMemcpyHostToDevice));//type of the DNA in the box
    free(chromatinIndex);
    free(chromatinStart);
    free(chromatinType);

	CoorBasePair *StraightChrom = (CoorBasePair*)malloc(sizeof(CoorBasePair)*STRAIGHT_BP_NUM);
	fname = document["straightChromatin"].GetString();
	printf("Straight Chromatin Table: Reading %s\n", fname.c_str());
	FILE *fpStraight = fopen(fname.c_str(),"r");
	int dump;
	float bx, by, bz, rx, ry, rz, lx, ly, lz;
    for (int i=0; i<STRAIGHT_BP_NUM; i++)
	{
	    fscanf(fpStraight,"%d %f %f %f %f %f %f %f %f %f\n", &dump, &bx, &by, &bz, &rx, &ry, &rz, &lx, &ly, &lz);
	    //if(i<5) printf("%d %f %f %f %f %f %f %f %f %f\n", dump, bx, by, bz, rx, ry, rz, lx, ly, lz);
		StraightChrom[i].base.x = bx;
		StraightChrom[i].base.y = by;
		StraightChrom[i].base.z = bz;
		StraightChrom[i].right.x = rx;
		StraightChrom[i].right.y = ry;
		StraightChrom[i].right.z = rz;
		StraightChrom[i].left.x = lx;
		StraightChrom[i].left.y = ly;
		StraightChrom[i].left.z = lz;
	}
	fclose(fpStraight);
	CUDA_CALL(cudaMalloc((void**)&dev_straightChrom, STRAIGHT_BP_NUM * sizeof(CoorBasePair)));
	CUDA_CALL(cudaMemcpy(dev_straightChrom, StraightChrom, STRAIGHT_BP_NUM * sizeof(CoorBasePair), cudaMemcpyHostToDevice));

	CoorBasePair *BendChrom = (CoorBasePair*)malloc(sizeof(CoorBasePair)*BEND_BP_NUM);
	fname = document["bentChromatin"].GetString();
	printf("Bend Chromatin Table: Reading %s\n", fname.c_str());
	FILE *fpBend = fopen(fname.c_str(),"r");
    for (int i=0; i<BEND_BP_NUM; i++)
	{
	    fscanf(fpBend,"%d %f %f %f %f %f %f %f %f %f\n", &dump, &bx, &by, &bz, &rx, &ry, &rz, &lx, &ly, &lz);
	    //if(i<5) printf("%d %f %f %f %f %f %f %f %f %f\n", dump, bx, by, bz, rx, ry, rz, lx, ly, lz);
		BendChrom[i].base.x = bx;
		BendChrom[i].base.y = by;
		BendChrom[i].base.z = bz;
		BendChrom[i].right.x = rx;
		BendChrom[i].right.y = ry;
		BendChrom[i].right.z = rz;
		BendChrom[i].left.x = lx;
		BendChrom[i].left.y = ly;
		BendChrom[i].left.z = lz;
	}
	fclose(fpBend);
	CUDA_CALL(cudaMalloc((void**)&dev_bendChrom, BEND_BP_NUM * sizeof(CoorBasePair)));
	CUDA_CALL(cudaMemcpy(dev_bendChrom, BendChrom, BEND_BP_NUM * sizeof(CoorBasePair), cudaMemcpyHostToDevice));
	
	float hisx, hisy, hisz;
	float3* bendHistone = (float3*)malloc(sizeof(float3)*BEND_HISTONE_NUM);
	fname = document["bentHistone"].GetString();
	printf("Bent Histone Table: Reading %s\n", fname.c_str());
	FILE *fpBentH = fopen(fname.c_str(),"r");
    for (int i=0; i<BEND_HISTONE_NUM; i++)
	{
	    fscanf(fpBentH,"%f %f %f\n", &hisx, &hisy, &hisz);
	    //if(i<5) printf("%f %f %f\n", hisx, hisy, hisz);
		bendHistone[i].x = hisx;
		bendHistone[i].y = hisy;
		bendHistone[i].z = hisz;
	}
	fclose(fpBentH);
	CUDA_CALL(cudaMalloc((void**)&dev_bendHistone, BEND_HISTONE_NUM * sizeof(float3)));
	CUDA_CALL(cudaMemcpy(dev_bendHistone, bendHistone, BEND_HISTONE_NUM * sizeof(float3), cudaMemcpyHostToDevice));
	
	float3 *straightHistone = (float3*)malloc(sizeof(float3)*STRAIGHT_HISTONE_NUM);
	fname = document["straightHistone"].GetString();
	printf("Straight Histone Table: Reading %s\n", fname.c_str());
	FILE *fpStraiH = fopen(fname.c_str(),"r");
    for (int i=0; i<STRAIGHT_HISTONE_NUM; i++)
	{
	    fscanf(fpStraiH,"%f %f %f\n", &hisx, &hisy, &hisz);
	    //if(i<5) printf("%f %f %f\n", hisx, hisy, hisz);
		straightHistone[i].x = hisx;
		straightHistone[i].y = hisy;
		straightHistone[i].z = hisz;
	}
	fclose(fpStraiH);
	CUDA_CALL(cudaMalloc((void**)&dev_straightHistone, STRAIGHT_HISTONE_NUM * sizeof(float3)));
	CUDA_CALL(cudaMemcpy(dev_straightHistone, straightHistone, STRAIGHT_HISTONE_NUM * sizeof(float3), cudaMemcpyHostToDevice));
	
	free(StraightChrom);
	free(BendChrom);	
	free(bendHistone);	
	free(straightHistone);

	printf("DNA geometry has been loaded to GPU memory\n");	 
	int* tmp=(int*) malloc(sizeof(int)*27);
	int kk=0;	
	for(int iz = -1; iz < 2; iz ++)
    {
        for(int iy = -1; iy < 2; iy ++)
        {
	        for(int ix = -1; ix < 2; ix ++)
			{
				tmp[kk] = iz * NUCLEUS_DIM * NUCLEUS_DIM + iy * NUCLEUS_DIM + ix;
				//printf("idx_neig = %d, iz = %d, iy = %d, iz = %d, h_deltaidxBin_neig = %d\n", idx_neig, iz, iy, ix, tmp[idx_neig]);
				kk++;
			}
		}
	}
	CUDA_CALL(cudaMemcpyToSymbol(neighborindex,tmp,sizeof(int)*27,0,cudaMemcpyHostToDevice));
	free(tmp);
	printf("Finish initialize neighborindex\n");

	CUDA_CALL(cudaMemcpyToSymbol(d_rDNA,rDNA,sizeof(float)*12,0,cudaMemcpyHostToDevice));
	if(verbose>1)
	{
		for(int i=0;i<12;i++)
		{
			printf("radius %f\n",rDNA[i]);
		}
	}

	printf("Setting for judging DNA damage");
	float tmpf =-14.5238;
	CUDA_CALL(cudaMemcpyToSymbol(min1,&tmpf,sizeof(float),0,cudaMemcpyHostToDevice));
	tmpf =-14.4706;
	CUDA_CALL(cudaMemcpyToSymbol(min2,&tmpf,sizeof(float),0,cudaMemcpyHostToDevice));
	tmpf =-32.0530;
	CUDA_CALL(cudaMemcpyToSymbol(min3,&tmpf,sizeof(float),0,cudaMemcpyHostToDevice));
	tmpf =14.5238;
	CUDA_CALL(cudaMemcpyToSymbol(max1,&tmpf,sizeof(float),0,cudaMemcpyHostToDevice));
	tmpf =14.4706;
	CUDA_CALL(cudaMemcpyToSymbol(max2,&tmpf,sizeof(float),0,cudaMemcpyHostToDevice));
	tmpf =31.8126;
	CUDA_CALL(cudaMemcpyToSymbol(max3,&tmpf,sizeof(float),0,cudaMemcpyHostToDevice));
}

__device__ float caldistance(float3 pos1, float3 pos2)
{
	return (sqrtf((pos1.x -pos2.x)*(pos1.x -pos2.x)+(pos1.y -pos2.y)*(pos1.y -pos2.y)+(pos1.z -pos2.z)*(pos1.z -pos2.z)));
}

__device__ float3 pos2local(int type, float3 pos, int index)
{
//do the coordinate transformation, index is the linear index for the referred box
//from global XYZ to local XYZ so that we can use the position of DNA base in two basic type (Straight and Bend) 
	int x = index%NUCLEUS_DIM;//the x,y,z index of the box
	int z = floorf(index/(NUCLEUS_DIM*NUCLEUS_DIM));
	int y = floorf((index%(NUCLEUS_DIM*NUCLEUS_DIM))/NUCLEUS_DIM);
	
	pos.x = pos.x-(2*x + 1 - NUCLEUS_DIM)*UNITLENGTH*0.5;//relative to its center
	pos.y = pos.y-(2*y + 1 - NUCLEUS_DIM)*UNITLENGTH*0.5;
	pos.z = pos.z-(2*z + 1 - NUCLEUS_DIM)*UNITLENGTH*0.5;

	float xc, yc, zc; // local coordinates in locally oriented coordinates frame
	switch(type)
	{
	//Straight type
	case 1:
		{xc = pos.x;
		yc = pos.y;
		zc = pos.z;
		break;}
	case 2://-z
		{xc = -pos.x;//Ry(pi)
		yc = pos.y;
		zc = -pos.z;
		break;}
	case 3://+y
		{xc = pos.x;//Rx(pi/2)
		yc = -pos.z;
		zc = pos.y;
		break;}
	case 4:
		{xc = pos.x;
		yc = pos.z;
		zc = -pos.y;
		break;}
	case 5://+x
		{xc = -pos.z;//Ry(-pi/2)
		yc = pos.y;
		zc = pos.x;
		break;}
	case 6:
		{xc = pos.z;
		yc = pos.y;
		zc = -pos.x;
		break;}
	case 7://Bend
		{xc = pos.x;
		yc = pos.y;
		zc = pos.z;
		break;}
	case 8:
		{xc = -pos.z;//Rz(pi)Ry(pi/2)
		yc = -pos.y;
		zc = -pos.x;
		break;}
	case 9:
		{xc = -pos.x;//Rz(pi)
		yc = -pos.y;
		zc = pos.z;
		break;}
	case 10:
		{xc = -pos.z;//Ry(-pi/2)
		yc = pos.y;
		zc = pos.x;
		break;}
	case 11:
		{xc = -pos.x;//Ry(pi)
		yc = pos.y;
		zc = -pos.z;
		break;}
	case 12:
		{xc = pos.z;//Rz(pi)Ry(-pi/2)
		yc = -pos.y;
		zc = pos.x;
		break;}
	case 13:
		{xc = pos.x;//Rx(pi)
		yc = -pos.y;
		zc = -pos.z;
		break;}
	case 14:
		{xc = pos.z;//Ry(pi/2)
		yc = pos.y;
		zc = -pos.x;
		break;}
	case 15:
		{xc = pos.y;//Rz(-pi/2)
		yc = -pos.x;
		zc = pos.z;
		break;}
	case 16:
		{xc = -pos.z;//Ry(-pi/2)Rz(pi/2)
		yc = pos.x;
		zc = -pos.y;
		break;}
	case 17:
		{xc = -pos.y;//Rz(pi/2)
		yc = pos.x;
		zc = pos.z;
		break;}
	case 18:
		{xc = -pos.z;//Rz(-pi/2)Rx(pi/2)
		yc = -pos.x;
		zc = pos.y;
		break;}
	case 19:
		{xc = pos.y;//Rz(-pi/2)Ry(pi)
		yc = pos.x;
		zc = -pos.z;
		break;}
	case 20:
		{xc = pos.z;//Rz(-pi/2)Rx(-pi/2)
		yc = -pos.x;
		zc = pos.y;
		break;}
	case 21:
		{xc = -pos.y;//Rz(pi/2)Ry(pi)
		yc = -pos.x;
		zc = -pos.z;
		break;}
	case 22:
		{xc = pos.z;//Rz(pi/2)Rx(pi/2)
		yc = pos.x;
		zc = pos.y;
		break;}
	case 23:
		{xc = pos.x;//Rx(pi/2)
		yc = -pos.z;
		zc = pos.y;
		break;}
	case 24:
		{xc = -pos.y;//Rz(pi/2)Ry(pi/2)
		yc = pos.z;
		zc = -pos.x;
		break;}
	case 25:
		{xc = -pos.x;//Rx(pi/2)Ry(pi)
		yc = pos.z;
		zc = pos.y;
		break;}
	case 26:
		{xc = -pos.y;//Rx(pi/2)Rz(pi/2)
		yc = -pos.z;
		zc = pos.x;
		break;}
	case 27:
		{xc = pos.x;//Rx(-pi/2)
		yc =pos.z;
		zc = -pos.y;
		break;}
	case 28:
		{xc = pos.y;//Rx(pi/2)Rz(-pi/2)
		yc = -pos.z;
		zc = -pos.x;
		break;}
	case 29:
		{xc = -pos.x;//Rx(-pi/2)Ry(pi)
		yc = -pos.z;
		zc = -pos.y;
		break;}
	case 30:
		{xc = pos.y;//Rz(-pi/2)Ry(-pi/2)
		yc = pos.z;
		zc = pos.x;
		break;}
	default:
	    {printf("wrong type\n");  // for test
		break;}
	}
	pos.x=xc;
	pos.y=yc;
	pos.z=zc;//*/
	return pos;
}



__global__ void chemSearch(int num, Edeposit* d_edrop, int* dev_chromatinIndex,int* dev_chromatinStart,int* dev_chromatinType, CoorBasePair* dev_straightChrom,
								CoorBasePair* dev_bendChrom,float3* dev_straightHistone,float3* dev_bendHistone, combinePhysics* d_recorde)
{
	int id = blockIdx.x*blockDim.x+ threadIdx.x;
	curandState localState = cuseed[id];
	float3 newpos, pos_cur_target;
	int3 index;
	CoorBasePair* chrom;
	float3 *histone;
	int chromNum, histoneNum,flag=0;
	while(id<num)
	{
		d_recorde[id].site.x=-1;//initialize
		d_recorde[id].site.y=-1;
		d_recorde[id].site.z=-1;
		d_recorde[id].site.w=-1;		
		d_recorde[id].prob2=d_edrop[id].e;
		d_recorde[id].prob1=curand_uniform(&localState);

		pos_cur_target=d_edrop[id].position;
		index.x=floorf((pos_cur_target.x+UNITLENGTH*NUCLEUS_DIM/2)/UNITLENGTH);
		index.y=floorf((pos_cur_target.y+UNITLENGTH*NUCLEUS_DIM/2)/UNITLENGTH);
		index.z=floorf((pos_cur_target.z+UNITLENGTH*NUCLEUS_DIM/2)/UNITLENGTH);

		int delta=index.x+index.y*NUCLEUS_DIM+index.z*NUCLEUS_DIM*NUCLEUS_DIM,minindex=-1;
		float distance[3]={100},mindis=100;
		for(int i=0;i<27;i++)
		{
			flag=0;
			int newindex = delta+neighborindex[i];
			if(newindex<0 || newindex > NUCLEUS_DIM*NUCLEUS_DIM*NUCLEUS_DIM-1) continue;
			int type = dev_chromatinType[newindex];
			if(type==-1 || type==0) continue;

			newpos = pos2local(type, pos_cur_target, newindex);
			if(type<7)
			{
				if(newpos.x<(min1-SPACETOBODER) || newpos.y<(min2-SPACETOBODER) || newpos.z<(min3-SPACETOBODER) ||newpos.x>(max1+SPACETOBODER)
				 || newpos.y>(max2+SPACETOBODER) || newpos.z>(max3+SPACETOBODER))
					continue;
				chrom=dev_straightChrom;
				chromNum=STRAIGHT_BP_NUM;
				histone=dev_straightHistone;
				histoneNum=STRAIGHT_HISTONE_NUM;
			}
			else
			{
				if(newpos.x<(min1-SPACETOBODER) || newpos.y<(min2-SPACETOBODER) || newpos.z<(min3-SPACETOBODER) ||newpos.x>(max3+SPACETOBODER)
				 || newpos.y>(max2+SPACETOBODER) || newpos.z>(max1+SPACETOBODER))
					continue;
				chrom=dev_bendChrom;
				chromNum=BEND_BP_NUM;
				histone=dev_bendHistone;
				histoneNum=BEND_HISTONE_NUM;
			}
			for(int j=0;j<histoneNum;j++)
			{
				mindis = caldistance(newpos, histone[j])-RHISTONE;
				if(mindis < 0) flag=1;
			}
			if(flag) break;
			for(int j=0;j<chromNum;j++)
			{
				// can take the size of base into consideration, distance should be distance-r;
				mindis=100,minindex=-1;
				distance[0] = caldistance(newpos, chrom[j].base)-RBASE;
				distance[1] = caldistance(newpos,chrom[j].left)-RSUGAR;
				distance[2] = caldistance(newpos,chrom[j].right)-RSUGAR;
				for(int iii=0;iii<3;iii++)
				{
					if(mindis>distance[iii])
					{
						mindis=distance[iii];
						minindex=iii;
					}
				}
				if(mindis<0)
				{
					if(minindex>0)
					{
						d_recorde[id].site.x = dev_chromatinIndex[newindex];
						d_recorde[id].site.y = dev_chromatinStart[newindex]+j;
						d_recorde[id].site.z = 3+minindex;
						d_recorde[id].site.w = 1;
					}
					flag=1;
					break;
				}
				int tmp = floorf(curand_uniform(&localState)/0.25);
				distance[0] = caldistance(newpos, chrom[j].base)-RBASE-d_rDNA[tmp];
				distance[1] = caldistance(newpos,chrom[j].left)-RSUGAR- d_rDNA[4];
				distance[2] = caldistance(newpos,chrom[j].right)-RSUGAR- d_rDNA[4];
				for(int iii=0;iii<3;iii++)
				{
					if(mindis>distance[iii])
					{
						mindis=distance[iii];
						minindex=iii;
					}
				}
				if(mindis<0)
				{
					if(minindex>0)
					{
						d_recorde[id].site.x = dev_chromatinIndex[newindex];
						d_recorde[id].site.y = dev_chromatinStart[newindex]+j;
						d_recorde[id].site.z = 3+minindex;
						d_recorde[id].site.w = 1;
					}
					flag=1;
					break;
				}
			}
			if(flag) break;
		}
		id+=blockDim.x*gridDim.x;
	}
	cuseed[id]=localState;
}

__global__ void phySearch(int num, Edeposit* d_edrop, int* dev_chromatinIndex,int* dev_chromatinStart,int* dev_chromatinType, CoorBasePair* dev_straightChrom,
								CoorBasePair* dev_bendChrom,float3* dev_straightHistone,float3* dev_bendHistone, combinePhysics* d_recorde)
{
	int id = blockIdx.x*blockDim.x+ threadIdx.x;
	curandState localState = cuseed[id];
	float3 newpos, pos_cur_target;
	int3 index;
	CoorBasePair* chrom;
	float3 *histone;
	int chromNum, histoneNum,flag=0;
	while(id<num)
	{
		d_recorde[id].site.x=-1;//initialize
		d_recorde[id].site.y=-1;
		d_recorde[id].site.z=-1;
		d_recorde[id].site.w=-1;		
		d_recorde[id].prob1=d_edrop[id].e;
		d_recorde[id].prob2=curand_uniform(&localState)*(EMAX-EMIN)+EMIN;

		pos_cur_target=d_edrop[id].position;
		index.x=floorf((pos_cur_target.x+UNITLENGTH*NUCLEUS_DIM/2)/UNITLENGTH);
		index.y=floorf((pos_cur_target.y+UNITLENGTH*NUCLEUS_DIM/2)/UNITLENGTH);
		index.z=floorf((pos_cur_target.z+UNITLENGTH*NUCLEUS_DIM/2)/UNITLENGTH);

		int delta=index.x+index.y*NUCLEUS_DIM+index.z*NUCLEUS_DIM*NUCLEUS_DIM,minindex=-1;
		float distance[3]={100},mindis=100;
		for(int i=0;i<27;i++)
		{
			flag=0;
			int newindex = delta+neighborindex[i];
			if(newindex<0 || newindex > NUCLEUS_DIM*NUCLEUS_DIM*NUCLEUS_DIM-1) continue;
			int type = dev_chromatinType[newindex];
			if(type==-1 || type==0) continue;

			newpos = pos2local(type, pos_cur_target, newindex);
			if(type<7)
			{
				if(newpos.x<(min1-SPACETOBODER) || newpos.y<(min2-SPACETOBODER) || newpos.z<(min3-SPACETOBODER) ||newpos.x>(max1+SPACETOBODER)
				 || newpos.y>(max2+SPACETOBODER) || newpos.z>(max3+SPACETOBODER))
					continue;
				chrom=dev_straightChrom;
				chromNum=STRAIGHT_BP_NUM;
				histone=dev_straightHistone;
				histoneNum=STRAIGHT_HISTONE_NUM;
			}
			else
			{
				if(newpos.x<(min1-SPACETOBODER) || newpos.y<(min2-SPACETOBODER) || newpos.z<(min3-SPACETOBODER) ||newpos.x>(max3+SPACETOBODER)
				 || newpos.y>(max2+SPACETOBODER) || newpos.z>(max1+SPACETOBODER))
					continue;
				chrom=dev_bendChrom;
				chromNum=BEND_BP_NUM;
				histone=dev_bendHistone;
				histoneNum=BEND_HISTONE_NUM;
			}
			for(int j=0;j<histoneNum;j++)
			{
				mindis = caldistance(newpos, histone[j])-RHISTONE;
				if(mindis < 0) flag=1;
			}
			if(flag) break;
			for(int j=0;j<chromNum;j++)
			{
				// can take the size of base into consideration, distance should be distance-r;
				mindis=100,minindex=-1;
				distance[0] = caldistance(newpos, chrom[j].base)-RBASE-RPHYS;
				distance[1] = caldistance(newpos,chrom[j].left)-RSUGAR- RPHYS;
				distance[2] = caldistance(newpos,chrom[j].right)-RSUGAR- RPHYS;
				for(int iii=0;iii<3;iii++)
				{
					if(mindis>distance[iii])
					{
						mindis=distance[iii];
						minindex=iii;
					}
				}
				if(mindis<0)
				{
					if(minindex>0)
					{
						d_recorde[id].site.x = dev_chromatinIndex[newindex];
						d_recorde[id].site.y = dev_chromatinStart[newindex]+j;
						d_recorde[id].site.z = 3+minindex;
						d_recorde[id].site.w = 0;
					}
					flag=1;
				}
			}
			if(flag) break;
		}
		//if(id%(blockDim.x*gridDim.x)==0) printf("id is %d\n", id);
		id+=blockDim.x*gridDim.x;//*/
	}
	cuseed[id]=localState;
}

void DNAList::run()
{
	std::string fname = document["fileForEnergy"].GetString();
	float threshold = document["compareEnergy"].GetFloat();
	float dep_sum;
    FILE *depofp = fopen(fname.c_str(), "r");
    if (depofp == NULL)
	{
		printf("The file 'deposit.txt' doesn't exist\n");
		dep_sum=0;
	}
	else
	{
		fscanf(depofp, "%f", &dep_sum);
		fclose(depofp);
	}
	if(dep_sum<threshold)
		return;
	
	int totalphy;
	fname = document["fileForTotalEvent"].GetString();
	Edeposit* edrop=readStage(&totalphy,0, fname.c_str());//read binary file x y z e
	printf("\n**********\ntotal initial number of physics energy deposit point is %d\n**********\n", totalphy);

	Edeposit* dev_edrop;
	cudaMalloc((void**)&dev_edrop,totalphy*sizeof(Edeposit));
	cudaMemcpy(dev_edrop, edrop, totalphy*sizeof(Edeposit), cudaMemcpyHostToDevice);
	free(edrop);
	
	combinePhysics* d_recorde;
	CUDA_CALL(cudaMalloc((void**)&d_recorde,sizeof(combinePhysics)*totalphy));

	phySearch<<<NRAND/256,256>>>(totalphy, dev_edrop, dev_chromatinIndex,dev_chromatinStart,dev_chromatinType, dev_straightChrom,
								dev_bendChrom, dev_straightHistone, dev_bendHistone, d_recorde);
	cudaDeviceSynchronize();
	CUDA_CALL(cudaFree(dev_edrop));

	combinePhysics* recorde=(combinePhysics*)malloc(sizeof(combinePhysics)*totalphy);		 
	CUDA_CALL(cudaMemcpy(recorde, d_recorde, sizeof(combinePhysics)*totalphy,cudaMemcpyDeviceToHost));

	chemReact* recordPhy= combinePhy(&totalphy, recorde,0);//consider the probability and generate final damage site
	printf("\n**********\neffective physics damage is %d\n**********", totalphy);
	free(recorde);
	CUDA_CALL(cudaFree(d_recorde));
/**************************************************************/
	int totalchem;
	fname = document["fileForChemPos"].GetString();
	Edeposit* chemdrop=readStage(&totalchem,1, fname.c_str());
	printf("\n**********\ntotal initial number of chemical  point is %d\n**********\n", totalchem);

	Edeposit* dev_chemdrop;
	cudaMalloc((void**)&dev_chemdrop,totalchem*sizeof(Edeposit));
	cudaMemcpy(dev_chemdrop, chemdrop, totalchem*sizeof(Edeposit), cudaMemcpyHostToDevice);
	free(chemdrop);
	combinePhysics* d_recordc;
	CUDA_CALL(cudaMalloc((void**)&d_recordc,sizeof(combinePhysics)*totalchem));

	chemSearch<<<NRAND/256,256>>>(totalchem, dev_chemdrop, dev_chromatinIndex,dev_chromatinStart,dev_chromatinType, dev_straightChrom,
								dev_bendChrom, dev_straightHistone, dev_bendHistone, d_recordc);
	cudaDeviceSynchronize();
	CUDA_CALL(cudaFree(dev_chemdrop));

	combinePhysics* recordc=(combinePhysics*)malloc(sizeof(combinePhysics)*totalchem);		 
	CUDA_CALL(cudaMemcpy(recordc, d_recordc, sizeof(combinePhysics)*totalchem,cudaMemcpyDeviceToHost));

	chemReact* recordChem= combinePhy(&totalchem, recordc,1);//consider the probability and generate final damage site
	printf("\n**********\neffective chemical damage is %d\n**********", totalchem);
	free(recordc);
	CUDA_CALL(cudaFree(d_recordc));

	if(totalphy+totalchem==0) {printf("NO DAMAGE AT ALL\n");return;}

	chemReact* totalrecord=(chemReact*)malloc(sizeof(chemReact)*(totalphy+totalchem));
	memcpy(totalrecord,recordPhy,sizeof(chemReact)*totalphy);
	memcpy(&totalrecord[totalphy],recordChem,sizeof(chemReact)*totalchem);
	free(recordPhy);
	free(recordChem);	    
	printf("total efective is %d\n**********", totalphy+totalchem);

	damageAnalysis(totalphy+totalchem,totalrecord,dep_sum/4.35568731e6,0,0);
	free(totalrecord);
}