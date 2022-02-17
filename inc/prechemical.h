#ifndef PRECHEMICAL_H
#define PRECHEMICAL_H
#include "global.h"

#define MAXNUMBRANCHPROD 3 // maximum number of products a branch can have
#define MAXBRANCHTYPE 6
#define MAXNUMBRANCH 3

#define PBRANCH2RECOMB 0.3 //the recombined H2O* deexcited to be H2O
#define PBRANCH11RECOMB 0.55 //0.55 // the recombined H2O* dissociative deexcited to be H. + OH.
#define PBRANCH12RECOMB 0.15 //0.15 // the recombined H2O* dissociative deexcited to be H2 + OH. + OH.
#define NTHREAD_PER_BLOCK_PAR 512

class PrechemList
{
public:
	PrechemList();
	~PrechemList();

	void readBranchInfo(std::string fname);
	void readThermRecombInfo(std::string fname); // loading the thermalization mean distance and recombination probability of the subexcitation electrons for prechemical stage simulation
	void readWaterStates();

	void initGPUVariables();
	void run();
	void saveResults();
// parameters for Branch Infor
	int nbtype; // number of all the different branch types
    int *num_product_btype; // the number of the products for each branch type
	int *ptype_product_btype; // the species type of the products for each branch type

	float *placeinfo_btype; // for each branch type, all the info (1 rms of hole hopping, two rms and coefficient for each product (1+2+2*3=9 entries for each branch) )
	
	int nb_rece; //number of branches for recombined electrons
	int nb_wi; //number of branches for ionized water molecule
    int nb_we_a1b1; //number of branches for excited water molecule with the excitation state to be A1B1
	int nb_we_b1a1; //number of branches for excited water molecule with the excitation state to be B1A1
	int nb_we_rd; // number of branches for excited water molecule with the excitation state to be Rydberg or diffusion bands
	int nb_w_dis; //number of branches for the dissociative water attachment
	
	float *pb_rece; //probability of each branch for recombined electrons
	float *pb_wi; //probability of each branch for ionized water molecule
	float *pb_we_a1b1;//probability of each branch for excited water molecule with the excitation state to be A1B1
	float *pb_we_b1a1; //probability of each branch for excited water molecule with the excitation state to be B1A1
	float *pb_we_rd; //probability of each branch for excited water molecule with the excitation state to be Rydberg or diffusion bands
	float *pb_w_dis; //probability of each branch for the dissociative water attachment	
	
	int *btype_rece; //the branch type of each branch for recombined electrons
	int *btype_wi; //the branch type of each branch for ionized water molecule
	int *btype_we_a1b1;//the branch type of each branch for excited water molecule with the excitation state to be A1B1
	int *btype_we_b1a1; //the branch type of each branch for excited water molecule with the excitation state to be B1A1
	int *btype_we_rd; //the branch type of each branch for excited water molecule with the excitation state to be Rydberg or diffusion bands
	int *btype_w_dis; //the branch type of each branch for the dissociative water attachment

// parameters for thermal relaxation
	float *p_recomb_elec; //energy-dependent recombination probability
    float *rms_therm_elec; //energy-dependent thermalization rms distance
    int nebin; //number of energy bins for the recombination probability and thermalization rms distance table
    float mine_ebin, ide_ebin; // minimum energy and inverse bin size of the energy bins

// parameters for information of waterradiolysis state
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
	int *wiid_elec; // the id of the parent ionized water molecule (for potential recombination) for the solvated electrons

// GPU variables
	cudaStream_t stream[5];

	float *d_posx, *d_posy, *d_posz; // the GPU variables to store the positions of the particles (a larger memory is required to include the product of prechemical stage) 
	float *d_ene, *d_ttime; // initial energies of the initial particles
	int *d_ptype, *d_index; // the species type of the particles (255 for empty entries or produced H2O)	
	int *d_statetype; // the statetype of the initial particles
	int *d_wiid_elec;// the parent ion id of electrons for potential recombination
};

#endif