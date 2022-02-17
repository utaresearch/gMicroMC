#include "DNAList.h"
compare_dnaindex compare1;
compare_baseindex compare2;
compare_boxindex compare3;
DNAList::DNAList()
{
	
}

DNAList::~DNAList()
{
	cudaFree(dev_bendChrom);
	cudaFree(dev_bendHistone);
	cudaFree(dev_straightChrom);
	cudaFree(dev_straightHistone);
	cudaFree(dev_chromatinIndex);
	cudaFree(dev_chromatinStart);
	cudaFree(dev_chromatinType);
}


void DNAList::calDNAreact_radius(float *diffCoef_spec)
{
	float k[12]={9,14,18,13,-10000,0, // hydrated electron
		6.1,9.2,6.4,6.1,1.8,0}; // hydroxyl
	for(int i=0;i<2;i++)
	{
		for(int j = 0;j<6;j++) // AGCT,sugar, histone
			rDNA[i*6+j]=k[i*6+j]/(7568.725*(diffCoef_spec[i]));//k 10^9 L/(mol*s), Diffusion 10^9 nm^2/ps. t ps
	}
}

Edeposit* DNAList::readStage(int *numPhy,int mode, const char* fname)
/*******************************************************************
c*    Reads electron reactive events from physics stage result     *
c*    Setup electron events as a list for the DNA damages          *
output *effphy 
Number of damage
c******************************************************************/
{
	int start,stop;
	float data[4];
	std::ifstream infile;
	if(mode==0) {infile.open(fname,std::ios::binary);printf("physics results: Reading %s\n",fname);}
	else {infile.open(fname,std::ios::binary);printf("physics results: Reading %s\n", fname);}
	
	start=infile.tellg();
    infile.seekg(0, std::ios::end);
    stop=infile.tellg();
    (*numPhy)=(stop-start)/16;
    if(*numPhy==0) {infile.close();return NULL;}
    infile.seekg(0, std::ios::beg);
	Edeposit *hs = (Edeposit*)malloc(sizeof(Edeposit)*(*numPhy));
	for(int i=0;i<(*numPhy);i++)
	{
		infile.read(reinterpret_cast <char*> (&data), sizeof(data));
		hs[i].position.x=data[0];
		hs[i].position.y=data[1];
		hs[i].position.z=data[2];
		if(mode==0) hs[i].e=data[3];
		else hs[i].e=1-PROBCHEM;
	}
	infile.close();
 	return hs;
}

void DNAList::quicksort(chemReact*  hits,int start, int stop, int sorttype)
{   
    //CPU sort function for ordering chemReacts in cpu memory
    if(stop == start) return;
    switch(sorttype)
    {
	    case 1:
	    {   
			std::sort(hits+start,hits+stop,compare1);
	        break;
	    }
	    case 2:
	    {   
			std::sort(hits+start,hits+stop,compare2);
	        break;
	    }
	    default:
	    {  
			std::sort(hits+start,hits+stop,compare1);
	        break;
	    }
    }
}
chemReact* DNAList::combinePhy(int* totalphy, combinePhysics* recorde,int mode)
{
	int counts=(*totalphy);
	std::sort(recorde,recorde+counts,compare3);
	
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
void DNAList::damageAnalysis(int counts, chemReact* recordpos,float totaldose,int totalPar,int totalOH)
{
	//if(counts==0) return;
	char buffer[256];
	
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
}

void DNAList::saveResults()
{
	std::string fname = document["fileForOutputDamage"].GetString();
	FILE* fp= fopen(fname.c_str(),"a");
    // fprintf(fp, "Dose SBcounts 0\n");
    // fprintf(fp, "%f\n", totaldose);
    fprintf(fp, "SSBd SSbi SSbm DSBd DSBi DSBm DSBh\n");
    for(int index=0;index<7;index++)
    	fprintf(fp, "%d ", results[index]);
    fprintf(fp, "\n");
    fprintf(fp, "SSB 2xSSB SSB+ 2SSB DSB DSB+ DSB++\n");
    for(int index=0;index<7;index++)
    	fprintf(fp, "%d ", complexity[index]);
   	fprintf(fp, "\n");
	fclose(fp);
}