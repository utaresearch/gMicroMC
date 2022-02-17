#ifndef PHYSICSLIST_H
#define PHYSICSLIST_H

#include <stdio.h>
#include <stdlib.h>

#include "global.h"

typedef struct
{ 
    gFloat x, y, z, elape_time;
    gFloat ux, uy, uz, e;
	int id, parentId;
	int ptype;

	int h2oState;
    int dead;
    gFloat path, A;
} Particle;

typedef struct
{
    gFloat x, y, z;
    gFloat e;
	int h2oState;
    gFloat time;
	int id;
	int parentId;
} Data;

class PhysicsList
{
public:
	PhysicsList();
	~PhysicsList();

	void readDataToGPU();
	void rd_pioncs();
	void rd_dacs(gFloat *DACSTable);
	void rd_ioncs(gFloat *ionBindEnergy, gFloat *ioncsTable);
	void rd_elast(gFloat *elastCSTable, gFloat *elastDCSTable);
	void rd_excit(gFloat *excitBindEnergy, gFloat *excitCSTable);

	void iniParticle();
	void outputParticle();

	void run();
	void saveResults();
	void transportParticles();

// private:
	int np, nWaitElec, sim_num;
	float max_e, totalTime;

	float NUCLEUS_RADIUS = 5500.0;
	int ContainerSize, MaxN, enumPerBatch, nElecPrimary, physicsPType;
	long long where_all=0;

	int *dev_where, *dev_second_num, *dev_gEid;
	Particle  *dev_pQueue, *dev_eQueue, *dev_e2Queue;
	Particle *h_particles;
	Data *dev_container, *h_container;

	cudaArray *dev_DACSTable, *dev_BindE_array, *dev_ieeCSTable, *dev_elastDCSTable;
	cudaArray *dev_protonTable;
	
	cudaChannelFormatDesc channelDesc;

	int INTERACTION_TYPES = 12, ODDS = 100, BINDING_ITEMS=5, DACS_OFFSET=43;
	int E_ENTRIES = 81, DACS_ENTRIES = 44;
	
	cudaTextureObject_t texObj_DACSTable    = 0,
						texObj_BindE_array  = 0,
						texObj_ieeCSTable   = 0,
						texObj_elastDCSTable = 0,
						texObj_protonTable    = 0;
};

#endif