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

	1. This code uses a mesh in `.face` and `.vert` format. Problems have been detected using msms.
	2. AMOEBA uses a `.key` file with the parameterization. The parameters can be included in `.key` file or redirect to a predefined parameter file like `amoebapro13.prm`, same as 1pgb example.
	3. You need a `.xyz` file with atom positions, the `.xyz` and `.key` filename must be the same.

# Multipoles_calculations_python
