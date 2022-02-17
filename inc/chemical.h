#ifndef CHEMICAL_H
#define CHEMICAL_H

#include "global.h"
#include "DNAList.h"

#include <vector>
using namespace std;

#define MAXPARTYPE 15  // maximum particle types
#define MAXREACTION 25  // maximum types of reactions
#define MAXNUMREACTANT4PAR 10 // maximum types of reactants that a particle can react with 
#define MAXNUMREACTANT4REACT 3 // maximum number of reactant a reaction may have
#define MAXNUMNEWPAR4REACT 3 // maximum number of new particles generated in a reaction
#define NUMDIFFTIMESTEPS 5 // number of different time steps used during chemistry stage
#define MAXNUMNZBIN 300000 //maximum number of non-zero bins
#define MAXNUMBIN 25000000000 // maximum number of bins
#define MAXNUMTAGBIN (MAXNUMBIN/32) //use 1 bit to denote one bin, use unsigned int (32 bit) to store this tag data of non-zero bin 
#define MAXCANDIDATE 100 // maximal number of the candidate reactions stored for each particle at the current time step
#define NUMOUTPUTMEM 20
#define NUMOXYGEN 0

class ChemList
{
public:
	ChemList();
	~ChemList();
	void readRadiolyticSpecies();

	void readReactions();
	void setReactantList();
	void setReactRadius();

	void readIniRadicals();
	void cleanRadicals();

	void iniGPU();
	void copyDataToGPU();

	void run(DNAList ddl);

	void saveNvsTime();
	void saveResults();

public:
//parameters for reactants
	int numSpecType;
    float *diffCoef_spec, *radii_spec;
	float maxDiffCoef_spec;
	std::vector<std::string> Name_spec;
	float diffTemCoef = 1;

// parameters for reactions
	int numReact;	// number of reactions
	int numReactant_React[MAXREACTION], indexReactant_React[MAXREACTION + 1], typeReactant_React[MAXREACTION * MAXNUMREACTANT4REACT]; // parameters about reactants	
	int numNewPar_React[MAXREACTION], indexNewPar_React[MAXREACTION + 1], typeNewPar_React[MAXREACTION * MAXNUMNEWPAR4REACT];// parameters about the new particles generated in the reactions	
	float kobs_React[MAXREACTION], radii_React[MAXREACTION], prob_React[MAXREACTION], diffCoef_total_React[MAXREACTION];// chemistry characteristic about the reactions
	
	int numReactant_Par[MAXPARTYPE];// parameters arranged in the order of each particle type to look for possible neighbors that can arise a reaction
	int typeReactant_Par[MAXPARTYPE * MAXNUMREACTANT4PAR], subtypeReact_Par[MAXPARTYPE * MAXNUMREACTANT4PAR];

	float h_deltaT_adap[NUMDIFFTIMESTEPS]; // five different time steps to be adaptively used during the chemistry stage
	float calc_radii_React[MAXREACTION * NUMDIFFTIMESTEPS]={0};	
	float max_calc_radii_React[NUMDIFFTIMESTEPS]={0};
	float reactTemCoef[21] = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
  
// for radicals
	float *posx, *posy, *posz, *ttime;
	int *index;
	unsigned char *ptype;
	int iniPar, maxPar;

//global
	int numCurPar; // total number of current live particles;
	float *h_mintd_Par_init;
	int h_deltaidxBin_neig[27];
	const int NTHREAD_PERBLOCK_CHEM = 256;
	float curTime = 1, recordInterval = 1, elapseRecord = 0, reactInterval = 1, elapseReact = 0;
	float4* recordposition;

// GPU variables
	float *d_posx, *d_posy, *d_posz; //variables to store the original particle position for each time step and store the generated new particles at the end of the time step
	float *d_posx_d, *d_posy_d, *d_posz_d; // variables to store the new particle positions of d_posx, d_posy, d_posz after diffusion
	float *d_posx_s, *d_posy_s, *d_posz_s; // variables to store the sorted particle positions of d_posx, d_posy, d_posz
	float *d_posx_sd, *d_posy_sd, *d_posz_sd; // variables to store the sorted particle positions of d_posx_d, d_posy_d, d_posz_d

	float *d_ttime;
	int *d_index; // these two labels are not used now to distinguish different radicals, there is no need to label _d, _s, _sd

	unsigned char *d_ptype, *d_ptype_s; //since for diffusion step, particle type doesn't change, hence no need to have d_ptype_d and d_ptype_sd

	unsigned char *d_statusPar; // 0: live; 255: dead particles, had reactions to generate new particles; 1: new particles generated in reactions at current time step.

	int *d_gridParticleIndex;
	unsigned long *d_gridParticleHash;

	int *d_accumParidxBin; // particle idx start pointer for each bin
	unsigned long *d_nzBinidx; // bin index of the non-zero bins containing particles
	//int *d_numParBinidx; // number of particles for the non-zero bins
	int *d_idxnzBin_neig; // index of the 27 neighboring bins containing particles within the non-zero bins: -1: no particles inside this neighboring bin; nonnegative value: the index within the non-zero bins
	int *d_idxnzBin_numNeig;

	float *d_mintd_Par;
	float4* d_recordposition;
};


#endif