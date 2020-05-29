#ifndef MICROGOMC_GLOBAL_H_
#define MICROGOMC_GLOBAL_H_


#include <vector_types.h>

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


#define MAXNUMPAR 131072 //1048576 //524288 // maximum particles at one time
#define MAXNUMPAR2 MAXNUMPAR*3 //maximum particles to be stored on GPU (current particles including dead ones and new ones in a reaction)
#define MAXNUMNZBIN 2000000 //maximum number of non-zero bins

//parameters for DNA damage analysis
#define EMIN 17.5
#define EMAX 17.5
#define PROBCHEM 0.4
#define RPHYS 0.1
#define dDSB 10
#define dS 216

#define MAXPARTYPE 15  // maximum particle types
#define MAXREACTION 25  // maximum types of reactions

#define MAXNUMREACTANT4PAR 10 // maximum types of reactants that a particle can react with 
#define MAXNUMREACTANT4REACT 3 // maximum number of reactant a reaction may have

#define MAXNUMNEWPAR4REACT 3 // maximum number of new particles generated in a reaction

#define MAXNUMBIN 25000000000 // maximum number of bins
#define MAXNUMTAGBIN (MAXNUMBIN/32) //use 1 bit to denote one bin, use unsigned int (32 bit) to store this tag data of non-zero bin 
#define NUMDIFFTIMESTEPS 5 // number of different time steps used during chemistry stage

#define MAXCANDIDATE 100 // maximal number of the candidate reactions stored for each particle at the current time step
#define NUMOUTPUTMEM 20
#define NUMSTEPSPEROUTPUT 50

#define NTHREAD_PER_BLOCK_PAR 256
#define NTHREAD_PER_BLOCK_BIN 64

#define PI 3.1415926535897932384626433

//#define PROCESS_TIME 2.3e3  // 1ns
/******************************************************************************************************
*******************************************************************************************************
******************************************************************************************************/
#define NUCLEUS_DIM 200 //# of bins
#define STRAIGHT_BP_NUM 5040
#define BEND_BP_NUM 4041
#define BEND_HISTONE_NUM 24
#define STRAIGHT_HISTONE_NUM 30
#define UNITLENGTH 55
__constant__  int neighborindex[27]={0};
__constant__ float min1=-14.5238, min2 =-14.4706, min3 = -32.0530, max1 = 14.5238, max2 = 14.4706, max3=31.8126;



#define DiffusionOfOH 2.8 //  10^9 nm*nm/s
#define SPACETOBODER 2
#define RBASE 0.5
#define RHISTONE 3.13
#define RSUGAR 0.431

#define CUDA_CALL(x) do{if(x != cudaSuccess){printf("CUDA Error at %s:%d\n",__FILE__,__LINE__);return;}}while(0)

typedef struct
{ 
    float3 base, right, left;
} CoorBasePair;

typedef struct
{ 
    int index;
    int boxindex;
    float3 position;
    float3 dir;
} DNAsegment;

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

__constant__  float d_rDNA[6]={0};
chemReact* recordposition;
chemReact* d_recordposition;
__device__ int d_totalspace=25000;
int totalspace=25000;
float rDNA[6];

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
compare_dnaindex compare1;
compare_baseindex compare2;
compare_boxindex compare3;
/******************************************************************************************************
*******************************************************************************************************
******************************************************************************************************/

class ChemistrySpec
{
public:
	ChemistrySpec();
	~ChemistrySpec();

	void initChemistry(string fileprefix);
	void readRadiolyticSpecies(string fname);

public:
    int numSpecType;
    float *diffCoef_spec, *radii_spec;
	float maxDiffCoef_spec;
	std::vector<std::string> Name_spec;
};

class ReactionType
{
public:
	ReactionType();
	~ReactionType();

	void initReaction(ChemistrySpec chemistrySpec, string fileprefix);
	void readReactionTypeInfo(ChemistrySpec chemistrySpec, string fname);
	void setReactantList_Par(ChemistrySpec chemistrySpec);
	void calcReactionRadius(ChemistrySpec chemistrySpec);

public:
	// parameters arranged in the order of each reactions
	int numReact;
	
	// parameters about reactants
	int numReactant_React[MAXREACTION], indexReactant_React[MAXREACTION + 1], typeReactant_React[MAXREACTION * MAXNUMREACTANT4REACT];
	
	// parameters about the new particles generated in the reactions
	int numNewPar_React[MAXREACTION], indexNewPar_React[MAXREACTION + 1], typeNewPar_React[MAXREACTION * MAXNUMNEWPAR4REACT];
	
	// chemistry characteristic about the reactions
	float kobs_React[MAXREACTION], radii_React[MAXREACTION], prob_React[MAXREACTION], diffCoef_total_React[MAXREACTION];

	// parameters arranged in the order of each particle type to look for possible neighbors that can arise a reaction
	int numReactant_Par[MAXPARTYPE];
	int typeReactant_Par[MAXPARTYPE * MAXNUMREACTANT4PAR], subtypeReact_Par[MAXPARTYPE * MAXNUMREACTANT4PAR];
	
	float h_deltaT_adap[NUMDIFFTIMESTEPS]; // five different time steps to be adaptively used during the chemistry stage
	float calc_radii_React[MAXREACTION * NUMDIFFTIMESTEPS];
	
	float max_calc_radii_React[NUMDIFFTIMESTEPS];
};

class ParticleData
{
public:
	ParticleData();
	~ParticleData();

	void readInitialParticles_RITRACK(string fname);
    void readInitialParticles_GEANT4(string fname);
	
public:
	int initnumPar; 
	float initTime;

	float *posx, *posy, *posz, *ttime;
	int *index;
	unsigned char *ptype;
    
	int converTable[MAXPARTYPE];
};

//gpu variables from ChemistrySpec class
__device__ __constant__ float d_diffCoef_spec[MAXPARTYPE];
__device__ __constant__ float d_radii_spec[MAXPARTYPE];
__device__ __constant__ float d_maxDiffCoef_spec[1];

//gpu variables from ReactionType class
__device__ int d_numReactant_React[MAXREACTION];
__device__ int d_indexReactant_React[MAXREACTION + 1];
__device__ int d_typeReactant_React[MAXREACTION * MAXNUMREACTANT4REACT];
__device__ int d_numNewPar_React[MAXREACTION];
__device__ int d_indexNewPar_React[MAXREACTION + 1];
__device__ int d_typeNewPar_React[MAXREACTION * MAXNUMNEWPAR4REACT];
__device__ int d_numReactant_Par[MAXPARTYPE];
__device__ int d_typeReactant_Par[MAXPARTYPE * MAXNUMREACTANT4PAR];
__device__ int d_subtypeReact_Par[MAXPARTYPE * MAXNUMREACTANT4PAR];	
__device__ float d_kobs_React[MAXREACTION];
__device__ float d_calc_radii_React[MAXREACTION * NUMDIFFTIMESTEPS];
__device__ float d_prob_React[MAXREACTION];

//gpu variables from ParticleData class
float *d_posx, *d_posy, *d_posz, *d_ttime; //variables to store the original particle position for each time step and store the generated new particles at the end of the time step
                                 // therefore the size needs to be 3 times of the number of particles to store new particles
float *d_posx_d, *d_posy_d, *d_posz_d; // variables to store the new particle positions of d_posx, d_posy, d_posz after diffusion
float *d_posx_s, *d_posy_s, *d_posz_s; // variables to store the sorted particle positions of d_posx, d_posy, d_posz
float *d_posx_sd, *d_posy_sd, *d_posz_sd; // variables to store the sorted particle positions of d_posx_d, d_posy_d, d_posz_d
int *d_index;
unsigned char *d_ptype, *d_ptype_s; //since for diffusion step, particle type doesn't change, hence no need to have d_ptype_d and d_ptype_sd

texture<float,1,cudaReadModeElementType> posx_tex;
texture<float,1,cudaReadModeElementType> posy_tex;
texture<float,1,cudaReadModeElementType> posz_tex;
texture<unsigned char,1,cudaReadModeElementType> ptype_tex;

texture<float,1,cudaReadModeElementType> posx_d_tex;
texture<float,1,cudaReadModeElementType> posy_d_tex;
texture<float,1,cudaReadModeElementType> posz_d_tex;
	
//gpu global variables used during simulation
unsigned char *d_statusPar; // 0: live; -1: dead particles, had reactions to generate new particles; 1: new particles generated in reactions at current time step.
int *d_gridParticleIndex;
unsigned long *d_gridParticleHash;
int *d_accumParidxBin; // particle idx start pointer for each bin
unsigned long *d_nzBinidx; // bin index of the non-zero bins containing particles
int *d_numParBinidx; // number of particles for the non-zero bins
int *d_idxnzBin_neig; // index of the 27 neighboring bins containing particles within the non-zero bins: -1: no particles inside this neighboring bin; nonnegative value: the index within the non-zero bins
int *d_idxnzBin_numNeig;

__device__ int d_deltaidxBin_neig[27];
int h_deltaidxBin_neig[27];

float *d_mintd_Par, *h_mintd_Par_init;

__device__ int d_numNewPar[1]; // the total number of new particles generated at current 
__device__ float d_deltaT[1];

int iniPar,iniCurPar;
int numCurPar; // total number of current live particles;

float curTime = 1.0f;
float h_deltaT;

float *output_posx, *output_posy, *output_posz;
unsigned char *output_ptype;
FILE *fp_output, *fp_counter;

// seed to generate random numbers
int* iseed1_h;///[MAXNUMPAR2];
int* iseed1;//[MAXNUMPAR2];
__device__ curandState cuseed[MAXNUMPAR2];

//functions
void initGPUVariables(ChemistrySpec *chemistrySpec, ReactionType *reactType, ParticleData *parData);
void runMicroMC(ChemistrySpec *chemistrySpec,ReactionType *reactType, ParticleData* parData, int process_time, int flagDNA);
void inirngG(int value);

void *outputParticles(void *idx_iter);

//kernel functions
__global__ void addOxygen(int num, unsigned char* d_ptype, int* d_index, float* d_posx, float* d_posy,
				float* d_posz, float* d_ttime, int numCurPar, float minx, float maxx,
				float miny, float maxy, float minz, float maxz);
__global__ void changePtype(unsigned char* d_ptype, int numCurPar, int target_value);
__global__ void assignBinidx4Par(unsigned long   *d_gridParticleHash,  // output
               int *d_gridParticleIndex, // output
               float *d_posx,               // input: positions
			   float *d_posy,
			   float *d_posz,
			   const float min_posx,     // input: minimal positions
			   const float min_posy, 
			   const float min_posz, 
			   unsigned long numBinx, // number of bins in x dimension
			   unsigned long numBiny, 
			   unsigned long numBinz,
			   const float binSize,
               int    numCurPar);
			 							   
__global__ void FindParIdx4NonZeroBin(unsigned long *d_gridParticleHash,
                                      unsigned long *d_nzBinidx,                           
                                      int *d_accumParidxBin, 
									  int numNZBin,
									  int numCurPar);	
									  
__global__ void FindNeig4NonZeroBin(unsigned long *d_nzBinidx, 
                                    int *d_idxnzBin_neig,
                                    int *d_idxnzBin_numNeig,									
									int numNZBin);									  
			   
__global__ void reorderData_beforeDiffusion(float *d_posx_s,        // output: sorted positions
							                float *d_posy_s,        // output: sorted positions
							                float *d_posz_s,        // output: sorted positions
							                unsigned char *d_ptype_s, // output: sorted positions				
                                            int   *gridParticleIndex,// input: sorted particle indices
                                            int    numCurPar);	 
			   
__global__ void reorderData_afterDiffusion(float *d_posx_s,        // output: sorted positions
							               float *d_posy_s,        // output: sorted positions
							               float *d_posz_s,        // output: sorted positions
							               unsigned char *d_ptype_s, // output: sorted positions
							               float *d_posx_sd, 
							               float *d_posy_sd, 
							               float *d_posz_sd, 
                                           int   *gridParticleIndex,// input: sorted particle indices
                                           int    numCurPar);	   
							
__global__ void react4TimeStep_beforeDiffusion(float *posx_new, 
                                               float *posy_new, 
								               float *posz_new, 
								               unsigned char *ptype_new,
                                               unsigned long *gridParticleHash,
                                               int *d_idxnzBin_neig,
											   int *d_idxnzBin_numNeig,
                                               unsigned long *d_nzBinidx,
                                               int *d_accumParidxBin,
								               unsigned char *d_statusPar,
								               float *d_mintd_Par,
								               unsigned long numBinx, // number of bins in x dimension
			                                   unsigned long numBiny, 
			                                   unsigned long numBinz,
								               int numNZBin,
								               int numCurPar,
								               int idx_typeTimeStep);

__global__ void react4TimeStep_afterDiffusion(float *posx_new, 
                                               float *posy_new, 
								               float *posz_new, 
								               unsigned char *ptype_new,
                                               unsigned long *gridParticleHash,
                                               int *d_idxnzBin_neig,
											   int *d_idxnzBin_numNeig,
                                               unsigned long *d_nzBinidx,
                                               int *d_accumParidxBin,
								               unsigned char *d_statusPar,
								               unsigned long numBinx, // number of bins in x dimension
			                                   unsigned long numBiny, 
			                                   unsigned long numBinz,
								               int numNZBin,
								               int numCurPar,
								               int idx_typeTimeStep);											   

__global__ void makeOneJump4Diffusion(float *d_posx_d, 
                                      float *d_posy_d, 
								      float *d_posz_d,                               
								      int numCurPar);								
__global__ void setupcuseed(int* iseed1);	
								
// functions used within the kernel function
__device__ int findBinIdxInNonZeroBinArray(unsigned long binidx, unsigned long *d_nzBinidx, int numNZBin);

__device__ unsigned int giveNonZeroTag4Bins(unsigned long binidx, unsigned int *d_tagnzBin);

__device__ int search4Reactant_beforeDiffusion(int idx_in_nzbinidx, 
                                                 int *d_accumParidxBin, 
												 unsigned char *d_statusPar, 
												 float *posx_new, 
												 float *posy_new, 
												 float *posz_new, 
												 unsigned char *ptype_new,
											     float *d_mintd_Par,
												 curandState *localState_pt, 
												 int numCurPar,
												 int idx_typeTimeStep);
												 
__device__ int search4Reactant_afterDiffusion(int idx_in_nzbinidx, 
                                                 int *d_accumParidxBin, 
												 unsigned char *d_statusPar, 
												 float *posx_new, 
												 float *posy_new, 
												 float *posz_new, 
												 unsigned char *ptype_new,
												 curandState *localState_pt, 
												 int numCurPar,
												 int idx_typeTimeStep);												 
	
__device__ int generateNewPar(int reactType, 
                              float calc_radii,
                              float dis_par_target_neig, 
							  int ptype_target, 
							  int ptype_neig, 
							  int idx_par_neig, 		 
							  float3 *pos_target, 
							  float3 *pos_neig, 
							  unsigned char *d_statusPar, 
							  float *posx_new, 
							  float *posy_new, 
							  float *posz_new, 
							  unsigned char *ptype_new,
							  curandState *localState_pt,
							  int numCurPar);	
							  
__device__ void rotate(float3 &direction, float costh, float phi);

__device__ void myswap(int *vec, const int i, const int j);
__device__ int mypermute_onecomb(int *vec, int istart, int iend, int radnum, int *times_permute);

		
#endif
