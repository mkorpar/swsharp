# DEPENDENCIES

## WINDOWS*:
    1) CUDA SDK 5.0
    2) Visual Studio 2010 (optional)

## LINUX and MAC OS:
Application uses following software:
    1) Linux shell
    2) gcc 4.4 or (for newest nvcc) - depends on nvcc version
    3) nvcc 2.*+
    4) doxygen - for documentation generation (optional)
    5) mpi - for swsharpdbmpi (optional)

*note: The Visual Studio project provided by this distribution is only tested
with the listed software. There is no guarantee it could be run on other 
software setups.

# INSTALLATION

## LINUX and MAC OS:
Makefile is provided in the project root folder. If mpi is available uncomment
the swsharpdbmpi module on the top of the Makefile. After running make and all
dependecies are satisfied, include, lib and bin folders will appear. All 
executables are located in the bin folder. Exposed swsharp core api is located 
in the include folder, and swsharp core static library is found in the lib 
folder.

## WINDOWS:
Download the executables from https://sourceforge.net/projects/swsharp/files/ or
download the Visual Studio project from the same location.

# NOTES

Individual README files for executables are available in folders of the same 
name as the executable. 
