# gMicroMC
***
Revised on Apr. 9, 2020 by Youfang Lai.  
Corresponding email: xun.jia@utsouthwestern.edu, yujie.chi@uta.edu  
The details of the simulation package can be referred to 
* [1](https://aapm.onlinelibrary.wiley.com/action/showCitFormats?doi=10.1002%2Fmp.14037) Tsai, Min‐Yu, Zhen Tian, Nan Qin, Congchong Yan, Youfang Lai, Shih‐Hao Hung, Yujie Chi, and Xun Jia. 
"A new open‐source GPU‐based microscopic Monte Carlo simulation tool for the calculations of DNA damages caused by ionizing radiation--
Part I: Core algorithm and validation." Med. Phys., 47: 1958-1970 (2020). doi:10.1002/mp.14037
* [2](https://iopscience.iop.org/article/10.1088/1361-6560/aa6246/pdf) Tian, Zhen, Steve B. Jiang, and Xun Jia. "Accelerated Monte Carlo simulation on the chemical stage in water radiolysis using GPU." 
Physics in Medicine & Biology 62, 8:3081 (2017).
***
## Hardware and software requirements
A GPU that supports CUDA should be installed previously.
Program has been tested in Nvidia Quadro P400, Nvidia Titan Xp and Nvidia V100 GPU cards.
Ubuntu 16.04 and 18.04 have been tested. CUDA versions of 9.0, 9.2, 10.0, 10.2 have been tested.

## Code structure
The code is composed of the following three sequential parts. The users can run them separately.  
1. Part I --“phy_stage” for the electron physical transport
Input files: simulation configuration file “config.txt”, and source configuration file “source.txt”.  
In the “config.txt” file, four components are defined, including the GPU device index, the cutoff energy for the electron transport, 
the file name for the source configuration and one idle input for possible future use. Function readConfig() is responsible for it.    
In the “source.txt” file, the source is configured. Below we show the example in detail and explain it. The users should refer to 
the iniElectron() function to bettern understand it or change the logic of read in files.  
1 0  
4500.0  
0.00065  
0 0 0 0 0 0 1 4500  
0 0 0 0 0 0 1 4500  
0 0 0 0 0 0 1 4500  
The first line: Electron simulation history N and a flag (0 and 1) to specify two different configurations. Specifically,  
   * Flag = 0, all electrons will be initialized with a kinetic energy E (specified in line 2), within a sphere of radius R (specified in line 3). Their positions and velocity directions will be randomly sampled. 
   * Flag = 1, the initial state of the N electrons will be specified in sequential. For each line (starting from line 4) representing one electron, quantities are defined in the order of position (x, y, z), global time t, normalized velocity direction (vx, vy, vz) and kinetic energy (E).
   Notice, in this situation, lines 2-3 are required to be kept, but the information will not be used.  
   
   The second line: kinetic energy (in eV)  
   The third line: spherical radius (in cm)  
   The fourth to N+3 lines: each lines contain position (x, y, z, in cm), global time t (in seconds), normalized velocity direction (vx, vy, vz) and kinetic energy E (in eV).  
2. Part II – “prechem_stage” for the de-excitation of water molecules
No configuration file is needed. Probabilities of different channels and recombination of sub-electrons are given in the ./Input folder. It will automatically use the physint.dat and physfloat.dat in ../phys/output folder.
Caution: May need to change MAXNUMPAR in the “microMC_prechem_global.h” file to make it larger than the water states generated in the phy_stage. However, the over estimation of MAXNUMPAR  will slow down the entire simulation.
3. Part III – “chem_stage” for the radical difussion in the chemical stage and the DNA damage analysis
No config file is needed. Information of reaction lists and radical properties are given in the ./Input folder. DNA structure, along with the parameters to analyze the DNA damage are given in the ./table folder. DNA geometry is always included in the 5.5 um sphere centered at (0,0,0).
Caution: May need to change MAXNUMPAR in the “microMC_chem.h” file to make it larger than the number of radicals generated in the prechem_stage. Over estimation of MAXNUMPAR  will slow down the entire simulation.  
Parameters for DNA damage analysis are listed in microMC_chem.h too. Usually the default one is enough to have a comparable result. But it may need to be changed.  
Output fot the chemical stage is inside the ./Results folder. Change the output style in runMicroMC.cu.
## How to run the program
Because og the limit of file size in Github, one data file is compressed. PLease unzip the zip file in ./chem_stage/table to avoid possible errors.  
Use “./compile_cuMC” to compile each part of the program. May need “chmod + x ./compile_cuMC” to make it executable.  
Part I: use “./microMC config.txt” to run the program.  
Part II: use “./prechem 0” to make the program running on GPU 0.  
Part III: two scenarios. 1) use “./chem 0 1000 0” to run the program on GPU 0, with chemical stage ending at 1000 ps, and w/o DNA damage analysis starting. 2) use “./chem 0 1000 1” to run the program on GPU 0, with chemical stage ending at 1000 ps and w/ DNA damage analysis starting. Remember to copy the “totalphy.dat” in the ./phy_stage/output folder into the ./chem_stage/Results folder before the execution if these three steps are ran separately.

To make the logic of the simulation smooth, a “main.c” file outside the three folders is created. After defining parameters in the “source.txt” and the “config.txt” files,
just use “gcc main.c -o gMicroMC” for the compilation and use “./gMicroMC” for the execution of the entire three parts.
