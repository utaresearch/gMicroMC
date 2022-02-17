#ifndef DNALIST_H
#define DNALIST_H

#include "global.h"
#include <algorithm>

#define NUCLEUS_DIM 200 //# of bins
#define STRAIGHT_BP_NUM 5040
#define BEND_BP_NUM 4041
#define BEND_HISTONE_NUM 24
#define STRAIGHT_HISTONE_NUM 30
#define UNITLENGTH 55 // size of a voxel in nm

#define EMIN 17.5
#define EMAX 17.5
#define PROBCHEM 0.4

#define DiffusionOfOH 2.8 //  10^9 nm*nm/s
#define SPACETOBODER 2
#define RBASE 0.5
#define RHISTONE 3.13
#define RSUGAR 0.431
#define RPHYS 0.1
#define dDSB 10
#define dS 216

typedef struct
{ 
    float3 base, right, left;
} CoorBasePair;

typedef struct
{ 
    float e;
    float3 position;
} Edeposit;

typedef struct
{
	int x, y, z, w;//DNA index, base index, left or right, damage type
}chemReact;

typedef struct
{
	chemReact site;
	float prob1,prob2;
}combinePhysics;

class DNAList
{
public:
    DNAList();
    ~DNAList();
    void initDNA();
    void saveResults();
    void calDNAreact_radius(float* diffCoeff);
    Edeposit* readStage(int *numPhy,int mode, const char* fname);
    void quicksort(chemReact*  hits,int start, int stop, int sorttype);
    chemReact* combinePhy(int* totalphy, combinePhysics* recorde,int mode);
    void damageAnalysis(int counts, chemReact* recordpos,float totaldose,int idle1,int idle2);
    void run();
public:
    float rDNA[12]={0};
    int complexity[7]={0,0,0,0,0,0,0};//SSB,2xSSB, SSB+, 2SSB, DSB, DSB+, DSB++
	int results[7]={0,0,0,0,0,0,0};//SSBd, SSbi, SSbm, DSBd, DSBi, DSBm, DSBh.
    //GPU
    CoorBasePair *dev_bendChrom, *dev_straightChrom;
    float3 *dev_bendHistone, *dev_straightHistone;
    int *dev_chromatinIndex, *dev_chromatinStart, *dev_chromatinType;

};

struct compare_dnaindex
{
    __host__ __device__ bool operator()(chemReact a, chemReact b)
    {
        return a.x < b.x;
    }
};
struct compare_baseindex
{
    __host__ __device__ bool operator()(chemReact a, chemReact b)
    {
        return a.y < b.y;
    }
};
struct compare_boxindex
{
    __host__ __device__ bool operator()(combinePhysics a, combinePhysics b)
    {
        return a.site.x < b.site.x;
    }
};

#endif