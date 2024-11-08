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

### Simulation Parameters
- **Nr, Na, Nz**: Number of grid points in radial, angular, and axial directions.
- **Lz**: Length of the domain in the z-direction.
- **sigma**: Ratio of cyclotron frequency to plasma frequency.
- **omega_n**: Driving frequency as a fraction of the plasma frequency.
- **envelope_1**: Gaussian envelope function to localize the source in the simulation.

### Output
The simulation outputs several fields (e.g., `Ez`, `Er`, `Ea`, `Bz`, `Br`, `Ba`) as HDF5 files. Each field represents a different component of the electric or magnetic field in the domain. Data files are stored in the specified output directory.

## Example
Below is an example of a command to start a simulation with 64 processes:
```bash
mpirun -np 64 python script.py
```

## Contact
For more information, please contact Xiuhong Xu at `xiuhong_xu@brown.edu`.

