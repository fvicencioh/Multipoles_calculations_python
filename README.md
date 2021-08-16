# Multipole Calculations

This code does molecular electrostatics calculations using a implicit Poisson-Boltzmann model and the boundary element method. The polarizable force field AMOEBA is also included and can be used to compute solvations energy consider the contribution of charges, dipoles and quadrupoles.
The following instructions assume that Ubuntu is the operating system.

## How to dowload

Create a clone of the repository:

	> cd $HOME
	> git clone https://github.com/Multipoles_calculations_python.git

### Creating the environment

Once the repository is in your machine, create the environment to use multipole calculations:

	> cd $HOME
	> cd Multipoles_calculations_python
	> conda env create -f conda.yml

This will install the following dependencies:

	* Python
	* Numpy
	* Numba
	* Scipy
	* Pyopencl
	* ExaFMM
	* Bempp-cl

Then, activate the environment with:

	> conda activate bempp_prod
    
#### Additional dependencies

This code can generate meshs using NanoShaper software, but trimesh library is required, it can be installed with:

    > conda install -c conda-forge trimesh
    
Inside the environment.

#### Using the code

To use the code, first activate the environment:

	> cd $HOME
	> cd Multipoles_calculations_python
	> conda activate bempp_prod

Then, use Python of Jupyter Notebooks to import the function:

	> import multipole_calculations
	> from multipole_calculations.multipoles import multipole_calculations

More details of the required and optional arguments in the sample file

	> Example_1pgb

##### Files required

Generate a folder with the `.key` and `.xyz` using the same file names. The code will automatically create intermediate files (mesh, `.xyzr`). The `.xyzr` file will be stored in this same folder, the mesh will be stored in:

	> multipole_calculations/ExternalSoftware/NanoShaper/meshs/filename/

