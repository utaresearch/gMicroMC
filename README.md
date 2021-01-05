# gMicroMC
***
This is the newest version of gMicroMC revised on Apr. 9, 2020 by Youfang Lai. It supports the simulation of physics stage from electron source and subsequent physicochemical and chemical stage for water radiolysis and radicals' reactions. Support for oxygen in chemcial stage was done and is waiting for validation. Support for proton source and concurrent transportation for DNA and radicals are being developed and validated. Redundant parts of the codes should be removed in the next version, which could help suppress the warning in compiling process. Unified data format for input and output in all three stages is expected to appear in the next verison as well.
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
Ubuntu 16.04 and 18.04 have been tested. 
CUDA versions of 9.0, 9.2, 10.0, 10.2 have been tested.

## Code structure
The code is composed of the following three sequential parts. The users can run them separately. Notice that the running of physicochemcial stage and chemical stage needs to read data from previous stage, namely physics stage and physicochemcial stage, respectively. The user may give their own designed data according to the data format needed by those two programs.

### Part I --“phy_stage” for the electron physical transport
Input files: simulation configuration file [phy_stage/config.txt](./phy_stage/config.txt), and source configuration file [phy_stage/source.txt](./phy_stage/source.txt).  
In the [phy_stage/config.txt](./phy_stage/config.txt) file, four components are defined, including the GPU device index, the cutoff energy for the electron transport, 
the file name for the source configuration and one idle input for possible future use. Function `readConfig()` is responsible for it.    
In the [phy_stage/source.txt](./phy_stage/source.txt) file, the source is configured. Below we show the example in detail and explain it. The users should refer to the `iniElectron()` function to bettern understand it or change the logic of read in files. 

```
1 0  
4500.0  
0.00065  
0 0 0 0 0 0 1 4500  
0 0 0 0 0 0 1 4500  
0 0 0 0 0 0 1 4500 
```

- The first line: Electron simulation history N and a flag (0 and 1) to specify two different configurations. Specifically,  
  - Flag = 0, all electrons will be initialized with a kinetic energy E (specified in line 2), within a sphere of radius R (specified in line 3). Their positions and velocity directions will be randomly sampled. 
  - Flag = 1, the initial state of the N electrons will be specified in sequential. For each line (starting from line 4) representing one electron, quantities are defined in the order of position (x, y, z), global time t, normalized velocity direction (vx, vy, vz) and kinetic energy (E).
   Notice, in this situation, lines 2-3 are required to be kept, but the information will not be used.  
   
- The second line: kinetic energy (in eV)  
- The third line: spherical radius (in cm)  
- The fourth to N+3 lines: each lines contain position (x, y, z, in cm), global time t (in seconds), normalized velocity direction (vx, vy, vz) and kinetic energy E (in eV).  
   
### Part II – “prechem_stage” for the de-excitation of water molecules
No configuration file is needed. Probabilities of different channels and recombination of sub-electrons are given in the [prechem_stage/Input](./prechem_stage/Input) folder. It will automatically use the `physint.dat` and `physfloat.dat` in `./phy_stage/output` folder.
Caution: May need to change `MAXNUMPAR` in the [prechem_stage/microMC_prechem_global.h](./prechem_stage/microMC_prechem_global.h) file to make it larger than the water states generated in the phy_stage. However, the over estimation of `MAXNUMPAR`  will slow down the entire simulation.

### Part III – “chem_stage” for the radical difussion in the chemical stage and the DNA damage analysis

No config file is needed. Information of reaction lists and radical properties are given in the [chem_stage/Input](./chem_stage/Input) folder. DNA structure, along with the parameters to analyze the DNA damage are given in the [chem_stage/table](./chem_stage/table) folder. DNA geometry is always included in the 5.5 um sphere centered at (0,0,0).
Caution: May need to change `MAXNUMPAR` in the [chem_stage/microMC_chem.h](./chem_stage/microMC_chem.h) file to make it larger than the number of radicals generated in the prechem_stage. Over estimation of `MAXNUMPAR`  will slow down the entire simulation.  
Parameters for DNA damage analysis are listed in [chem_stage/microMC_chem.h](./chem_stage/microMC_chem.h) too. Usually the default one is enough to have a comparable result. But it may need to be changed.  
Output fot the chemical stage is inside the [chem_stage/Results](./chem_stage/Results) folder. Change the output style in [chem_stage/runMicroMC.cu](./chem_stage/runMicroMC.cu).

## How to run the program
Because of the limit of file size in Github, one data file is compressed. PLease unzip the zip file in [chem_stage/table](./chem_stage/table) to avoid possible errors.  
Use `./compile_cuMC` to compile each part of the program. May need `chmod + x ./compile_cuMC` to make it executable.

```bash
# part I
# Notice, Inside the folder ./phy_stage run the program
./microMC config.txt

# Part II
# Notice, Inside the folder ./prechem_stage run the program
# make the program running on GPU 0
./prechem 0

# Part III
# Notice, Inside the folder ./chem_stage run the program
# copy the [phy_stage/output/totalphy.dat](./phy_stage/output/totalphy.dat) file into the [chem_stage/Results](./chem_stage/Results) 
cp ./phy_stage/output/totalphy.dat ./chem_stage/Results in case that the calculation of DNA damage is needed

# two scenarios
# 1. run the program on GPU 0, with chemical stage ending at 1000 ps, and w/o DNA damage analysis starting.
./chem 0 1000 0

# 2. run the program on GPU 0, with chemical stage ending at 1000 ps and w/ DNA damage analysis starting
./chem 0 1000 1
```

To make the logic of the simulation smooth, a [main.c](./main.c) file outside the three folders is created. After defining parameters in the “source.txt” and the “config.txt” files, just use `gcc main.c -o gMicroMC` for the compilation and use `./gMicroMC` for the execution of the entire three parts.
