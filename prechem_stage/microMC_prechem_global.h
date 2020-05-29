#ifndef MICROMC_PRECHEM_GLOBAL_H_
#define MICROMC_PRECHEM_GLOBAL_H_

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

#define MAXNUMPAR 262144//80000 // maximum particles at one time

#define NTHREAD_PER_BLOCK_PAR 256
#define NTHREAD_PER_BLOCK_BIN 64

#define PI 3.1415926535897932384626433
#define MAXNUMBRANCHPROD 3 // maximum number of products a branch can have
//#define CTHERMELEC 1.8 // rms per eV for electrons (nm/eV) 

#define MAXNUMPAR2 MAXNUMPAR*3 //maximum particles to be stored on GPU (current particles including dead ones and new ones in a reaction)
#define MAXBRANCHTYPE 6
#define MAXNUMBRANCH 3 //maximum number of dissociation branches for inoized or excited water

#define PBRANCH2RECOMB 0.3 //the recombined H2O* deexcited to be H2O
#define PBRANCH11RECOMB 0.55 //0.55 // the recombined H2O* dissociative deexcited to be H. + OH.
#define PBRANCH12RECOMB 0.15 //0.15 // the recombined H2O* dissociative deexcited to be H2 + OH. + OH.

#define MAXPVALUE 0.7357589  //2/e

typedef struct
{ 
	int parID;
	int ptype;
	int stype;
	float penergy;
	float parposX;
	float parposY;
	float parposZ;
	float etime;
} physicQ;

class ParticleData_prechem
{
public:
	ParticleData_prechem();
	~ParticleData_prechem();
	
    void readInitialParticles_GEANT4(string fname); // loading the initial particles for prechemical stage (geant4 simulation result)
	
public:
	int num_elec; //number of solvated electrons
	int num_wi; //number of ionized water molecule
	int num_we_a1b1; // number of excited water molecule with the excitation state to be A1B1
	int num_we_b1a1; // number of excited water molecule with the excitation state to be B1A1
	int num_we_rd; // number of excited water molecule with the excitation state to be Rydberg or diffusion bands
	int num_w_dis; // number of the dissociative water attachment
	int num_total; // total number of the electrons and water molecules
	
	int sidx_elec; // starting index for the solvated electrons in the particle array
	int sidx_wi; 
	int sidx_we_a1b1; 
	int sidx_we_b1a1; 
	int sidx_we_rd; 
	int sidx_w_dis; 
	
	float *posx, *posy, *posz; //current positions of the molecules
	float *ene, *ttime; // energy of the molecules
	
	int *statetype, *index; // the ionization state or excitation state of the water molecules (-1 for electrons)
	//float *parposx_elec, *parposy_elec, *parposz_elec; //parent positions of the electrons
	int *wiid_elec; // the id of the parent ionized water molecule (for potential recombination) for the solvated electrons
	
};

class Branch_water_prechem
{
public: 
    Branch_water_prechem();
	~Branch_water_prechem();
	
	void readBranchInfo(string fname); // loading the branching info for prechemical stage simulation
	
public:
    int *num_product_btype; // the number of the products for each branch type
	int *ptype_product_btype; // the species type of the products for each branch type
	float *placeinfo_btype; // for each branch type, all the info (1 rms of hole hopping, two rms and coefficient for each product (1+2+2*3=9 entries for each branch) )
	
	int nb_wi; //number of branches for ionized water molecule
    int nb_we_a1b1; //number of branches for excited water molecule with the excitation state to be A1B1
	int nb_we_b1a1; //number of branches for excited water molecule with the excitation state to be B1A1
	int nb_we_rd; // number of branches for excited water molecule with the excitation state to be Rydberg or diffusion bands
	int nb_w_dis; //number of branches for the dissociative water attachment
	
	float *pb_wi; //probability of each branch for ionized water molecule
	float *pb_we_a1b1;//probability of each branch for excited water molecule with the excitation state to be A1B1
	float *pb_we_b1a1; //probability of each branch for excited water molecule with the excitation state to be B1A1
	float *pb_we_rd; //probability of each branch for excited water molecule with the excitation state to be Rydberg or diffusion bands
	float *pb_w_dis; //probability of each branch for the dissociative water attachment
	
	int nbtype; // number of all the different branch types
	int *btype_wi; //the branch type of each branch for ionized water molecule
	int *btype_we_a1b1;//the branch type of each branch for excited water molecule with the excitation state to be A1B1
	int *btype_we_b1a1; //the branch type of each branch for excited water molecule with the excitation state to be B1A1
	int *btype_we_rd; //the branch type of each branch for excited water molecule with the excitation state to be Rydberg or diffusion bands
	int *btype_w_dis; //the branch type of each branch for the dissociative water attachment

};

class ThermRecomb_elec_prechem
{
public:
    ThermRecomb_elec_prechem();
	~ThermRecomb_elec_prechem();
	
	void readThermRecombInfo(string fname); // loading the thermalization mean distance and recombination probability of the subexcitation electrons for prechemical stage simulation
	
public:
    float *p_recomb_elec; //energy-dependent recombination probability
    float *rms_therm_elec; //energy-dependent thermalization rms distance
    int nebin; //number of energy bins for the recombination probability and thermalization rms distance table
    float mine_ebin, ide_ebin; // minimum energy and inverse bin size of the energy bins
};


//variables on GPU device

float *d_posx, *d_posy, *d_posz; // the GPU variables to store the positions of the particles (a larger memory is required to include the product of prechemical stage) 
float *d_ene, *d_ttime; // initial energies of the initial particles
int *d_ptype, *d_index; // the species type of the particles (255 for empty entries or produced H2O)	
int *d_statetype; // the statetype of the initial particles
int *d_wiid_elec;// the parent ion id of electrons for potential recombination

cudaArray *d_p_recomb_elec;
cudaArray *d_rms_therm_elec; 
texture<float,1,cudaReadModeElementType> p_recomb_elec_tex;	
texture<float,1,cudaReadModeElementType> rms_therm_elec_tex;	

__device__ __constant__ int d_num_elec; 
__device__ __constant__ int d_num_wi; 
__device__ __constant__ int d_num_we_a1b1; 
__device__ __constant__ int d_num_we_b1a1; 
__device__ __constant__ int d_num_we_rd; 
__device__ __constant__ int d_num_w_dis; 
__device__ __constant__ int d_num_total; 
	
__device__ __constant__ int d_sidx_elec; 
__device__ __constant__ int d_sidx_wi; 
__device__ __constant__ int d_sidx_we_a1b1; 
__device__ __constant__ int d_sidx_we_b1a1; 
__device__ __constant__ int d_sidx_we_rd; 
__device__ __constant__ int d_sidx_w_dis; 

__device__ __constant__ int d_nebin;
__device__ __constant__ float d_mine_ebin;
__device__ __constant__ float d_ide_ebin;

__device__ __constant__ int d_nbtype;
__device__ float d_placeinfo_btype[MAXBRANCHTYPE*9];
__device__ int d_num_product_btype[MAXBRANCHTYPE];
__device__ int d_ptype_product_btype[MAXBRANCHTYPE * MAXNUMBRANCHPROD];

__device__ __constant__ int d_nb_wi; 
__device__ __constant__ int d_nb_we_a1b1; 
__device__ __constant__ int d_nb_we_b1a1; 
__device__ __constant__ int d_nb_we_rd; 
__device__ __constant__ int d_nb_w_dis; 
	
__device__ float d_pb_wi[MAXNUMBRANCH]; 
__device__ float d_pb_we_a1b1[MAXNUMBRANCH]; 
__device__ float d_pb_we_b1a1[MAXNUMBRANCH];
__device__ float d_pb_we_rd[MAXNUMBRANCH];
__device__ float d_pb_w_dis[MAXNUMBRANCH];
 
__device__ int d_btype_wi[MAXNUMBRANCH]; 
__device__ int d_btype_we_a1b1[MAXNUMBRANCH]; 
__device__ int d_btype_we_b1a1[MAXNUMBRANCH];
__device__ int d_btype_we_rd[MAXNUMBRANCH];
__device__ int d_btype_w_dis[MAXNUMBRANCH];	
	
// seed to generate random numbers
int iseed1_h[MAXNUMPAR2];
__device__ int iseed1[MAXNUMPAR2];
__device__ curandState cuseed[MAXNUMPAR2];

//functions 
void initGPUVariables_pc(ParticleData_prechem *parData_pc, Branch_water_prechem *braInfo_pc, ThermRecomb_elec_prechem *thermRecombInfo_pc);
void runMicroMC_pc(ParticleData_prechem *parData_pc, Branch_water_prechem *braInfo_pc, ThermRecomb_elec_prechem *thermRecombInfo_pc);

__global__ void thermalisation_subexelectrons(float *d_posx, // x position of the particles (input and output)
                                              float *d_posy,
											  float *d_posz,
											  float *d_ene, // initial energies of the initial particles (input only)
											  int *d_ptype, // species type for products of prechemical stage, 255 for empty or produced water (output)
											  int *d_statetype, //the statetype of the initial particles (255 for died particles)
											  int *d_wiid_elec); // the index of the ionized water molecule for potential recombination

__global__ void dissociation_ionizedwater(float *d_posx,
                                          float *d_posy,
										  float *d_posz,
										  int *d_ptype,
										  int *d_statetype);			
										  
__global__ void dissociation_excitedwater_a1b1(float *d_posx,
                                               float *d_posy,
										       float *d_posz,
										       int *d_ptype,
										       int *d_statetype);	

__global__ void dissociation_excitedwater_b1a1(float *d_posx,
                                               float *d_posy,
										       float *d_posz,
										       int *d_ptype,
										       int *d_statetype);

__global__ void dissociation_excitedwater_rd(float *d_posx,
                                             float *d_posy,
										     float *d_posz,
										     int *d_ptype,
										     int *d_statetype);
											 
__global__ void dissociation_dissociativewater(float *d_posx,
                                               float *d_posy,
										       float *d_posz,
										       int *d_ptype,
										       int *d_statetype);												 

__device__ void getNormalizedDis_Sample3DGuassian(curandState *localState_pt, float *ndisx,float *ndisy,float *ndisz); //get a normalized 3D displacement by sampling a 3D guassian distribution   											  
__device__ void getDirection_SampleOnSphereSurface(curandState *localState_pt, float *ndisx, float *ndisy, float *ndisz); // get a random isotropic direction by sampling on a spheric surface                                                               					

__device__ void displace_twoproducts_noholehoping(float *d_posx, 
                                                  float *d_posy, 
												  float *d_posz,
												  curandState *localState_pt,
												  int btype, //branch type
												  int pid, // the current particle id
												  int pid_site); // the id of the particle considerred to be the original site (for recombination)
												  
__device__ void displace_threeproducts_noholehoping(float *d_posx, 
                                                  float *d_posy, 
												  float *d_posz, 
												  curandState *localState_pt,
												  int btype, //branch type
												  int pid, // the current particle id
												  int pid_site); // the id of the particle considerred to be the original site (for recombination)												  

__device__ void displace_twoproducts_holehoping(float *d_posx, 
                                                float *d_posy, 
												float *d_posz, 
												curandState *localState_pt,
												int btype, //branch type
												int pid); // the current particle id	

__device__ void displace_twoproducts_oneelec_holehoping(float *d_posx, 
                                                        float *d_posy, 
												        float *d_posz, 
												        curandState *localState_pt,
												        int btype, //branch type
												        int pid); // the current particle id													

__device__ void sampleThermalDistance(int pid, curandState *localState_pt, float *ndisx, float *ndisy, float *ndisz, float idx_ebin);												
//functions in the libGeometryRand.cu
void inirngG(int value);
__global__ void setupcuseed();

#endif
