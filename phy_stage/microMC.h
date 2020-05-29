#define DACS_OFFSET 43
#define BINDING_ITEMS 5
#define ODDS 100

#define DACS_ENTRIES 44
#define E_ENTRIES 81

#define ZERO 1.0e-20
#define SZERO 1.0e-6

#define REAL float
#define NPART 65536 //	number of particles simulated simultaneously

#define INTERACTION_TYPES 12
#define MCC 510998.9461 // rest mass energy ~511keV
#define TWOMCC 1021997.8922 //2*MC^2
#define M2C4 261119922915.31070521 //M^2C^4
#define PI 3.14159265359
#define C 299792458

#define GPUMEM_BASE_SIZE 300 // memory size in MB estimated for base needed
#define GPUMEM_PER_INC 4 // memory size in MB estimated per added particle

#define INC_2NDPARTICLES_100KEV 2500
#define INC_RADICALS_100KEV 12000

#define K_THREADS 128
#define REPORT_FILE 1
#define QSORT 3
#define NUCLEUS_RADIUS 5500 //100000000 // nm
#define E_DEPOSIT_FILE 1    // energy accumulation in a file.

//#define CURANDSTATE    curandStateXORWOW_t
#define CURANDSTATE  curandStateMRG32k3a_t // 3m13s
//#define CURANDSTATE curandStateXORWOW_t // 4m21s
//#define CURANDSTATE curandState_t

typedef struct
{ 
    REAL x, y, z;
    REAL ux, uy, uz;
    REAL e;
	int h2oState;
    int dead;
    REAL path;
	REAL elape_time;
	int id;
	int parentId;
} eStruct;
 
typedef struct
{
    REAL x, y, z;
    REAL e;
	int h2oState;
    REAL time;
	int id;
	int parentId;
} data;
