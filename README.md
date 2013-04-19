# SW# 

SW# (swsharp) is a library for sequence alignment based on CUDA enabled GPUs. It utilizes Hirschbergs and Ukkonens algorithm for memory efficiency and additional speed up. The library is scalable for use with multiple GPUs. Some parts of the library utilize MPI for CUDA enabled clusters.

## DEPENDENCIES

### WINDOWS

1. CUDA SDK 5.0
2. Visual Studio 2010 (optional)*

\*note: The Visual Studio project provided by this distribution is only tested with the listed software. There is no guarantee it could be run on other software setups.

### LINUX and MAC OS

Application uses following software:

1. gcc 4.*+
2. nvcc 2.*+
3. doxygen - for documentation generation (optional)
4. mpi - for swsharpdbmpi (optional)

## INSTALLATION

### LINUX and MAC OS
Makefile is provided in the project root folder. If mpi is available uncomment the swsharpdbmpi module on the top of the Makefile. After running make and all dependencies are satisfied, include, lib and bin folders will appear. All executables are located in the bin folder. Exposed swsharp core api is located in the include folder, and swsharp core static library is found in the lib folder. 

### WINDOWS
Download the executables from https://sourceforge.net/projects/swsharp/files/ or download the Visual Studio project from the same location.

### MODULES

Currently supported modules are:

1. swsharp - Module is the main static library used by other modules and does not provide any executable.
2. swsharpn - Module is used for aligning nucleotide sequnces.
3. swsharpp - Module is used for aligning protein sequnces.
4. swsharpnc - Module is used for aligning which searches the best scores on both strands of a nucleotide sequnces.

## NOTES

Individual README files for executables are available in folders of the same name as the executable. 
