#ifndef __REALTIME__
#define __REALTIME__
#include <stdio.h>
#include <algorithm>
#include <iostream>
#include <fstream>
#include "microMC_chem.h"

void printDevProp(int device)
//      print out device properties
{
    int devCount;
    cudaDeviceProp devProp;
//      device properties

    cudaGetDeviceCount(&devCount);
	cout << "Number of device:              " << devCount << endl;
	cout << "Using device #:                " << device << endl;
    cudaGetDeviceProperties(&devProp, device);
	
	printf("Major revision number:         %d\n",  devProp.major);
    printf("Minor revision number:         %d\n",  devProp.minor);
    printf("Name:                          %s\n",  devProp.name);
    printf("Total global memory:           %7.2f MB\n",  
	devProp.totalGlobalMem/1024.0/1024.0);
    printf("Total shared memory per block: %5.2f kB\n",  
	devProp.sharedMemPerBlock/1024.0);
    printf("Total registers per block:     %u\n",  devProp.regsPerBlock);
    printf("Warp size:                     %d\n",  devProp.warpSize);
    printf("Maximum memory pitch:          %u\n",  devProp.memPitch);
    printf("Maximum threads per block:     %d\n",  devProp.maxThreadsPerBlock);
    	
	printf("Maximum dimension of block:    %d*%d*%d\n", 			
	devProp.maxThreadsDim[0],devProp.maxThreadsDim[1],devProp.maxThreadsDim[2]);
	printf("Maximum dimension of grid:     %d*%d*%d\n", 
	devProp.maxGridSize[0],devProp.maxGridSize[1],devProp.maxGridSize[2]);
    printf("Clock rate:                    %4.2f GHz\n",  devProp.clockRate/1000000.0);
    printf("Total constant memory:         %5.2f kB\n",  devProp.totalConstMem/1024.0);
    printf("Texture alignment:             %u\n",  devProp.textureAlignment);
    printf("Concurrent copy and execution: %s\n",  (devProp.deviceOverlap ? "Yes" : "No"));
    printf("Number of multiprocessors:     %d\n",  devProp.multiProcessorCount);
    printf("Kernel execution timeout:      %s\n",  (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
//      obtain computing resource

}

void calDNAreact_radius(float* rDNA,float deltat)
{
	float k[5]={6.1,9.2,6.4,6.1,1.8};
	float tmp=sqrtf(PI*DiffusionOfOH*deltat*0.001);
	for(int i=0;i<5;i++)
	{
		rDNA[i]=k[i]/(4*PI*DiffusionOfOH)*10/6.023;//k 10^9 L/(mol*s), Diffusion 10^9 nm^2/s. t ps
		rDNA[i]=sqrtf(rDNA[i]*tmp+tmp*tmp*0.25)-tmp*0.5;
	}
	rDNA[5]=0;//histone protein absorption radius, assumed!!!
}

__device__ float caldistance(float3 pos1, float3 pos2)
{
	return (sqrtf((pos1.x -pos2.x)*(pos1.x -pos2.x)+(pos1.y -pos2.y)*(pos1.y -pos2.y)+(pos1.z -pos2.z)*(pos1.z -pos2.z)));
}

#if RANDGEO==0
__device__ float3 pos2local(int type, float3 pos, int index)
{
//do the coordinate transformation, index is the linear index for the referred box
//from global XYZ to local XYZ so that we can use the position of DNA base in two basic type (Straight and Bend) 
	int x = index%NUCLEUS_DIM;//the x,y,z index of the box
	int z = floorf(index/(NUCLEUS_DIM*NUCLEUS_DIM));
	int y = floorf((index%(NUCLEUS_DIM*NUCLEUS_DIM))/NUCLEUS_DIM);
	//printf("relative to type %d %d %d %d\n", type, x,y,z);
	pos.x = pos.x-(2*x + 1 - NUCLEUS_DIM)*UNITLENGTH*0.5;//relative to its center
	pos.y = pos.y-(2*y + 1 - NUCLEUS_DIM)*UNITLENGTH*0.5;
	pos.z = pos.z-(2*z + 1 - NUCLEUS_DIM)*UNITLENGTH*0.5;
	//printf("local coordinate %f %f %f\n", pos.x, pos.y, pos.z);

	float xc, yc, zc;
	switch(type)
	{
		//Straight type
	case 1:////!!!!!the following needs to be revised and confirmed
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
	curandState localState = cuseed[id%MAXNUMPAR2];
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
	cuseed[id%MAXNUMPAR2]=localState;
}

__global__ void phySearch(int num, Edeposit* d_edrop, int* dev_chromatinIndex,int* dev_chromatinStart,int* dev_chromatinType, CoorBasePair* dev_straightChrom,
								CoorBasePair* dev_bendChrom,float3* dev_straightHistone,float3* dev_bendHistone, combinePhysics* d_recorde)
{
	int id = blockIdx.x*blockDim.x+ threadIdx.x;
	curandState localState = cuseed[id%MAXNUMPAR2];
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
	cuseed[id%MAXNUMPAR2]=localState;
}//*/
#endif
/***********************************************************************************/

Edeposit* readStage(int *numPhy,int mode)
/*******************************************************************
c*    Reads electron reactive events from physics stage result     *
c*    Setup electron events as a list for the DNA damages          *
output *effphy 
Number of effective Physics damage
c******************************************************************/
{
	int start,stop;
	float data[4];
	ifstream infile;
	if(mode==0) {infile.open("./Results/totalphy.dat",ios::binary);printf("physics results: Reading ./Results/totalphy.dat\n");}
	else {infile.open("./Results/totalchem.dat",ios::binary);printf("physics results: Reading ./Results/totalchem.dat\n");}
	
	start=infile.tellg();
    infile.seekg(0, ios::end);
    stop=infile.tellg();
    (*numPhy)=(stop-start)/16;
    if(*numPhy==0) {infile.close();return NULL;}
    infile.seekg(0, ios::beg);
	Edeposit *hs = (Edeposit*)malloc(sizeof(Edeposit)*(*numPhy));
	for(int i=0;i<(*numPhy);i++)
	{
		infile.read(reinterpret_cast <char*> (&data), sizeof(data));
		if(i<8) printf("x y z e %f %f %f %f\n", data[0],data[1],data[2],data[3]);
		hs[i].position.x=data[0];
		hs[i].position.y=data[1];
		hs[i].position.z=data[2];
		if(mode==0) hs[i].e=data[3];
		else hs[i].e=1-PROBCHEM;
	}
	infile.close();
 	return hs;
}

void quicksort(chemReact*  hits,int start, int stop, int sorttype)
{   
    //CPU sort function for ordering chemReacts in cpu memory
    switch(sorttype)
    {
	    case 1:
	    {   sort(hits+start,hits+stop,compare1);
	        break;
	    }
	    case 2:
	    {   sort(hits+start,hits+stop,compare2);
	        break;
	    }
	    default:
	    {   sort(hits+start,hits+stop,compare1);
	        break;
	    }
    }
}
chemReact* combinePhy(int* totalphy, combinePhysics* recorde,int mode)
{
	int counts=(*totalphy);
	sort(recorde,recorde+counts,compare3);
	
	int j,num=0;
    for(int i=0; i<counts;)
    {
    	if(recorde[i].site.z==-1) {i++;continue;}
    	j=i+1;
        while(recorde[j].site.x==recorde[i].site.x)
        {
        	if(recorde[j].site.y==recorde[i].site.y && recorde[j].site.z==recorde[i].site.z)
        	{
        		if(mode==0) recorde[i].prob1 +=recorde[j].prob1;
        		else recorde[i].prob2 *= recorde[j].prob2;
        		recorde[j].site.z=-1;
        	}
        	j++;
        	if(j==counts) break;
        }        	
        i++;
    }
   
    for(int i=0;i<counts;i++)
    {
    	if(recorde[i].site.z!=-1 && recorde[i].prob2<recorde[i].prob1)
    	{
    		num++;
    	}
    }
    if(num==0) {(*totalphy)=0;return NULL;}

    chemReact* recordPhy=(chemReact*) malloc(sizeof(chemReact)*num);
    int index=0;
    for(int i=0;i<counts;i++)
    {
    	if(recorde[i].site.z!=-1 && recorde[i].prob2<recorde[i].prob1)
    	{
    		recordPhy[index].x=recorde[i].site.x;
    		recordPhy[index].y=recorde[i].site.y;
    		recordPhy[index].z=recorde[i].site.z;
    		recordPhy[index].w=recorde[i].site.w;
    		index++;
    	}
    }
    (*totalphy)=num;
    return recordPhy;
}
void damageAnalysis(int counts, chemReact* recordpos)
{
	// seems currently only the number of total SSB or DSB are correct
	// be careful to use the number in each category!!
	if(counts==0) return;
	char buffer[256];
	int complexity[7]={0};//SSB,2xSSB, SSB+, 2SSB, DSB, DSB+, DSB++
	int results[7]={0};//SSBd, SSbi, SSbm, DSBd, DSBi, DSBm, DSBh.
	
	quicksort(recordpos,0,counts,1);	
    int start=0,m,numofstrand,numoftype,k,cur_dsb;
    for(int i=0; i<counts;)
    {
    	if(recordpos[i].z==-1) {i++;continue;}
    	start=i;
        while(i<counts)
        {
        	if(recordpos[i].x==recordpos[start].x) i++;
        	else break;
        }
        if(i==start+1)//only one break on the DNA
        {
        	complexity[0]++;
        	results[recordpos[start].w]++;
        	continue;//find breaks in another DNA
        }

        if(i>start+1) quicksort(recordpos,start,i,2);//order damage sites so that search can be done
		cur_dsb=0;
        for(k=start;k<i-1;)//more than one break
        {
        	if(recordpos[k+1].y-recordpos[k].y>dS)
        	{
        		complexity[1]++;
        		results[recordpos[k].w]++;
        		k++;
        		continue;
        	}
        	else
        	{
	        	m=k+1;
	        	numoftype=0;
	        	numofstrand=0;
	        	int flag=0;//means SSB, 1 for DSB
        		while(m<i)
        		{
        			if( recordpos[m].z!=recordpos[m-1].z)//recordpos[m].y-recordpos[m-1].y<dDSB &&
        			{
        				numofstrand++;
        				if(recordpos[m].w!=recordpos[k].w) numoftype++;
        				int j=m;
        				int tmptype=0;
        				for(;j>k-1;j--)
        				{
        					if(recordpos[m].y-recordpos[j].y>dDSB) break;
        					if(recordpos[j].w!=recordpos[k].w) tmptype++;
        				}

        				if(j==k-1) flag=1;//DSB
        				else if(j==k && m==k+1) flag=2;//2SSB
        				else {m=j+1;numoftype-=tmptype;}
        				break;
        			}
        			if(recordpos[m].y-recordpos[k].y>dS) {m--;break;}//SSB+
        			if(recordpos[m].w!=recordpos[k].w) numoftype++;
    				m++;
        		}
        		if(flag==0)
        		{
        			complexity[2]++;
	        	 	if(numoftype!=0) results[2]++;
	        		else results[recordpos[k].w]++;//=m-k;
        		}
        		else if(flag==2)
        		{
        			complexity[3]++;
	        	 	if(numoftype!=0) results[2]++;
	        		else results[recordpos[k].w]++;
        		}
	        	else
	        	{//if flag=1,m must be k+1 and from k there must be a DSB
	        		m=k;//in consitent with the calculation of chem type,
	        		numoftype=0;
	        		int numofchem=0;
	        		while(m<i)
	        		{
	        			if(recordpos[m].y-recordpos[k].y<dDSB)
	        			{
	        				if(recordpos[m].w!=recordpos[k].w) numoftype++;
	        				if(recordpos[m].w==1) numofchem++;
	        				m++;
	        			}
	        			else
	        				break;
	        		}
	        		if(numofchem==1) results[6]++;
	        		else if(numoftype!=0) results[5]++;
	        		else results[3+recordpos[k].w]++;

	        		if(m-k==2) complexity[4]++;
	        		else complexity[5]++;
	        		cur_dsb++;
	        	}
	        	k=m;
        	}       	
        }
        if(cur_dsb>1) complexity[6]++;
        if(k==i-1)//deal with the last one in a segment
        {
        	complexity[1]++;
        	results[recordpos[k].w]++;
        }
    }

    FILE* fp= fopen("./Results/finalstat.txt","a");
    fprintf(fp, "SSBd SSbi SSbm DSBd DSBi DSBm DSBh\n");
    for(int index=0;index<7;index++)
    	fprintf(fp, "%d ", results[index]);
    fprintf(fp, "\n");
    fprintf(fp, "SSB 2xSSB SSB+ 2SSB DSB DSB+ DSB++\n");
    for(int index=0;index<7;index++)
    	fprintf(fp, "%d ", complexity[index]);
   	fprintf(fp, "\n");
	fclose(fp);//*/
}

#endif