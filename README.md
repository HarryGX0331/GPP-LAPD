# 3D Simulation of GPP-LAPD

## Overview
This project conducts a 3D simulation of the Gaseous Plasmon Polaritons (GPP) within the Large Plasma Device (LAPD) framework. Using Dedalus, this simulation solves a set of equations related to plasma dynamics in cylindrical coordinates, aiming to study the behavior of electric and magnetic fields under different conditions. The simulation runs in parallel using MPI for performance optimization on high-performance computing clusters.

## Features
- 3D simulation in cylindrical coordinates (z, theta, r)
- Adaptive time-stepping using the CFL condition
- Parallelized computation using MPI and Dedalus
- Customizable driving frequency and other plasma parameters
- Output data for electric and magnetic field components in HDF5 format

## Prerequisites
- Python 3.x
- [Dedalus](https://dedalus-project.readthedocs.io/en/latest/) (public and core modules)
- MPI for Python (`mpi4py`)
- `h5py` for handling output files
- `numpy` and `matplotlib` for data processing and visualization

## Installation
1. Install the required Python packages:
   ```bash
   pip install numpy h5py matplotlib mpi4py
   ```
2. Follow [Dedalus installation instructions](https://dedalus-project.readthedocs.io/en/latest/) for your environment.
3. Ensure MPI is configured correctly for parallel execution.

## Usage
To run the simulation, modify the `indices` and `base_output_path` variables as needed in the `main()` function.

Execute the script as follows:
```bash
mpirun -np <num_processes> python script.py
```
Replace `<num_processes>` with the desired number of MPI processes.

### Notes on MPI and Mesh

	-	The number of MPI processes must equal the product of the mesh dimensions specified in the code.
	-	Example: mesh = (8, 8) → -np 64
	-	The mesh dimensions are constrained by the grid sizes:
	   -	Na = 128
	   -	Nz = 128
	-	Each mesh dimension must:
	   1.	Be a factor of the corresponding grid size,
	   2.	Preferably be a power of two.
	


### Simulation Parameters
- **Nr, Na, Nz**: Number of grid points in radial, angular, and axial directions.
- **Lz**: Length of the domain in the z-direction.
- **sigma**: Ratio of cyclotron frequency to plasma frequency.
- **omega_n**: Driving frequency as a fraction of the plasma frequency.
- **envelope_1**: Gaussian envelope function to localize the source in the simulation.
- **omega_pnsq**: Density distributon along r direction.
- **k_z**: Fixed wavenumber along z direction.
- **density**: Peak density.

### Notes on the Units
- **Length**: normalized to $c / \omega_0\$, where $\omega_0\$ is a freely chosen plasma frequency.  
  For simplicity, in the code $\omega_0 = 1.5\text{GHz}\$.  
- **Time**: normalized to $1 / \omega_0\$.
- **Frequencies**: normalized to $\omega_0$. 

When **choosing $\omega_0$**, make sure it is appropriate for your simulation:  

- It should not be so small that reaching the desired physical time requires an excessively long normalized time.  
- It should not be so large that the corresponding length unit $c / \omega_0$ becomes too small.  
- Ideally, the **smallest physical distance of interest** should be larger than the grid spacing determined by $(Nr, Na, Nz)$.
### Output
The simulation outputs several fields (e.g., `Ez`, `Er`, `Ea`, `Bz`, `Br`, `Ba`) as HDF5 files. Each field represents a different component of the electric or magnetic field in the domain. Data files are stored in the specified output directory.
### Data Output Notes

The simulation uses Dedalus file handlers to save intermediate results.  
For example:

```python
check = ivp_solver.evaluator.add_file_handler(output_folder, iter=1, max_writes=500)
```
This prevents each HDF5 file from becoming too large.
If a single file is too large, reading it later (e.g., with MPI-parallel postprocessing) may fail or cause I/O errors.
By splitting output into smaller files, data is easier and safer to process.
## Example
Below is an example of a command to start a simulation with 64 processes:
```bash
mpirun -np 64 python /path/to/your/3D_S.py
```
### Submit with Slurm
Alternatively, here is an example Slurm batch script for running the same simulation on 64 processes:
```BASH
#!/bin/bash
#SBATCH -n 64
#SBATCH -t 48:00:00
#SBATCH --mem=128G
#SBATCH -J simulation_2.5GHz_1.2kG_2e13_kz_4cm_100ns
#SBATCH --output=/path/to/output/%A.out

mpiexec -n 64 python3 /path/to/your/3D_S.py
```
### Post-processing with data.py

After the simulation finishes, you can directly use data.py to read and extract the desired datasets from the generated HDF5 files:
```bash
#!/bin/bash
#SBATCH -n 10
#SBATCH -t 24:00:00
#SBATCH --mem=128G
#SBATCH -J S_data
#SBATCH --output=/path/to/output/%A_%a.out
#SBATCH --array=0-19 # Adjust this range according to how many simulation indices you have run in 3D_S.py

formatted_id=$SLURM_ARRAY_TASK_ID
mpiexec --mca orte_base_help_aggregate 0 --mca btl ^openib -n 10 python3 /path/to/data.py $formatted_id
```
## Contact
For more information, please contact Xiuhong Xu at `xx55@rice.edu`.

