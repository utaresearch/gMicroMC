# gMicroMC_v2.0
Updated by Youfang Lai. Email: youfanglai@gmail.com  
Cross-checked and finalized by Yujie Chi. Email: yujie.chi@uta.edu  
Correpondance: xun.jia@utsouthwestern.edu and yujie.chi@uta.edu  

Researchers are welcomed to modify and distribute this package for non-commercial purpose. The authors claim no responsibility for the results produced by users.
For credit recognition, please kindly cite
* Initial development  
Tsai, M. Y., Tian, Z., Qin, N., Yan, C., Lai, Y., Hung, S. H., ... & Jia, X. (2020). A new open‐source GPU‐based microscopic Monte Carlo simulation tool for the calculations of DNA damages caused by ionizing radiation‐‐‐Part I: Core algorithm and validation. Medical physics, 47(4), 1958-1970.
* Oxygen module  
Lai, Y., Jia, X., & Chi, Y. (2021). Modeling the effect of oxygen on the chemical stage of water radiolysis using GPU-based microscopic Monte Carlo simulations, with an application in FLASH radiotherapy. Physics in Medicine & Biology, 66(2), 025004.
* Proton, heavy ions and concurrent transport of DNA  
Lai, Y., Jia, X., & Chi, Y. (2021). Recent Developments on gMicroMC: Transport Simulations of Proton and Heavy Ions and Concurrent Transport of Radicals and DNA. International journal of molecular sciences, 22(12), 6615.

# Updated features
May 3rd, 2021  
1. Updated gMicroMC package with more comments and smoother control of the output by using class
2. All fuctions were divided into two types, kernel functions executed by GPU in \*.cu files and typical \*.cpp files

# Overview about the package
To make full use of this package and get meaningful results, users are suggested to well understand the general picture of microscopic MC simulation and the function of gMicroMC.
## 1）General picture of microscopic MC simulation
The microscopic MC simulation describes the process from the ionizing particle entering the cellular nucleus to the DNA damage formation. The entire process was decribed in three stages:
- physical stage, which describes the physical transport of the ionizing particles through a medium filled with water molecules, typically lasting from 10<sup>-15</sup> s to 10<sup>-12</sup> s. 
- physicochemical stage (sometimes referred to prechemical stage), which simulated the water radiolysis process with production of free chemical radicals.
- chemical stage, which deals with the diffusion and mutual ractions among chemical radicals, covering time scale from 10<sup>-12</sup> s to 10<sup>-6</sup> s. In our most recent development, oxygen molecules and DNA structures were also considered during this stage. 
- Finally, all the physical events and radical attack produce DNA damage, which are clustered into Double Strand Break (DSB) and Single Strand Break (SSB). 
Interested users are encouraged to read the following papers and the references therein for more details:
1. Friedland W, Dingfelder M, Kundrát P and Jacob P 2011 Track structures , DNA targets and radiation effects in the biophysical Monte Carlo simulation code PARTRAC Mutation Research - Fundamental and Molecular Mechanisms of Mutagenesis 711 28-40
2. Nikjoo H, Emfietzoglou D, Liamsuwan T, Taleei R, Liljequist D and Uehara S 2016 Radiation track , DNA damage and response — a review Reports on Progress in Physics 79 116601
3. Tsai M, Tian Z, Qin N, Yan C, Lai Y, Hung S-H, Chi Y and Jia X 2020 A new open-source GPU-based microscopic Monte Carlo simulation tool for the calculations of DNA damages caused by ionizing radiation --- Part I: Core algorithm and validation Med. Phys. 47 1958-70
4. Lai, Y., Jia, X., & Chi, Y. (2021). Modeling the effect of oxygen on the chemical stage of water radiolysis using GPU-based microscopic Monte Carlo simulations, with an application in FLASH radiotherapy. Physics in Medicine & Biology, 66(2), 025004.
5. Lai, Y., Jia, X., & Chi, Y. (2021). Recent Developments on gMicroMC: Transport Simulations of Proton and Heavy Ions and Concurrent Transport of Radicals and DNA. International journal of molecular sciences, 22(12), 6615.

## 2）What the package can provide
The advantage of using gMicroMC is to make a full use of GPU to accelerate the simulation process. It is very important as improving computational efficiency could enable more realistic yet computation-requiring simulations, for example, treating oxygen explicitly as molecules in the early age of chemical stage rather than viewing them as continuum background. The package does not introduce new physical or chemical interpretation. Hence, what gMicroMC can provide is basically the same as other CPU packages:
- Deposited energy, positions, track index etc. in the physics stage. (Check Data structure)
- Initial types of radicals and their distributions. (Check output from prechemical stage)
- Yields of different radicals at different moments, DNA damage sites in chemical stage.
- DNA strand break pattern after DNA damage grouping.
***

# Usage
## Structure
The code is structured as  
./ root folder  
&nbsp;&nbsp;&nbsp;&nbsp;./src/ --> source code  
&nbsp;&nbsp;&nbsp;&nbsp;./inc/ --> header files  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;./inc/rapidjson --> rapidjson library to deal with json files  
&nbsp;&nbsp;&nbsp;&nbsp;./tables/ --> predefined data for different process  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;./tables/physics --> physics cross sections  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;./tables/prechem --> info for decay channels and recombination of hydrated electrons  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;./tables/chem --> info for species and their reactions  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;./tables/dna --> stored dna structure in multiscale  
&nbsp;&nbsp;&nbsp;&nbsp;./output/ --> output directory   
## Compile
nvcc main.cpp ./src/* -o gMicroMC -I ./inc -rdc=true  
or  
make 

## Run the program
After compilation, the users needs to provide correct config.txt file, where illustration of parameters have been listed.  
Then running ./gMicroMC command will give you output files as defined in the config.txt.  


The work flow is  
prepare arrays like random seeds  
--> generating initial particles physicsList::h_particles  
--> simulate physics stage physicsList::run()  
--> store initial positions and types of water molecules physicsList::saveResults()  
--> reading prechemical stage PrechemList::readWaterStates()  
--> simulate prechemical stage and save results PrechemList::run()  
-->  reading chemical stage ChemList::d_posx, ChemList::d_posy, ChemList::d_posz, ChemList::d_ttime, ChemList::d_ptype, ChemList::d_index  
--> run chemical stage ChemList::run()  
--> save results ChemList::saveResults()  
--> read in for DNA damage analysis DNAList::posx  
--> simulated DNA damage analysis DNAList::run()  
--> DNA damage summary DNAList::saveResults();

All information is given in config.txt file, which is in json format.

Note
1. the saveResults() and readIn functions are not required mandatorily. Users can directly set the values in corresponding host arrays. For example, instead of physicsList::saveResults() and then prechemicalList::ReadIn(), users can directly set values of prechemicalList::posx, prechemicalList::posy, prechemicalList::posz, prechemicalList::ptype, prechemicalList::ttime, prechemicalList::index. This can be done in main.cpp fuction.
2. The reason why saveResults() fuction exists is due to the concern of memory. It is safer to save into files and then either read in by batchs or apply constraints to reduce the number of events (radicals).
3. The users have full freedom to change the data in ./tables folder, which could alter the defined physics interaction or decay channel. So be careful. 

