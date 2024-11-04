import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

print("Current version of Python is ", sys.version)
import os
import pathlib
import time
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import dedalus.public as d3
import dedalus.core as dec
from dedalus.tools import post
import logging
logger = logging.getLogger(__name__)

from dedalus.core.operators import GeneralFunction
from dedalus.extras import flow_tools
import shutil
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
if rank == 0:
    print(sys.version) # Print python version

# FILEHANDLER_TOUCH_TMPFILE = True
def run_simulation(index, output_folder):
    Nr = 64
    Na = 64
    Nz = 128
    #ratio = np.tanh(0.6*float(index)*0.1)
    ratio = np.tanh(0.6*float(index)*0.025)

    # Lz = 100
    Lz = 20

    sigma= 3.73333 # ratio of cyclotron frequency to plasma frequency
    omega_n = 2 # driving frequency as a fraction of omega_p, or eigenfrequency of GPP mode

    coords = d3.CartesianCoordinates('z','a','r')
    
    dist = d3.Distributor(coords, mesh=(8,8),  dtype= np.complex128)
    r_basis = d3.Chebyshev(coords['r'], Nr, bounds=[0.5, 3.75], dealias=1)
    theta_basis = d3.ComplexFourier(coords['a'], Na, bounds=[0, 2*np.pi], dealias=1)
    z_basis = d3.ComplexFourier(coords['z'], Nz, bounds=[0, Lz], dealias=1)

    domain = dec.domain.Domain(dist, bases=[z_basis,theta_basis,r_basis])
    V = dist.VectorField(coordsys=coords, name='V', bases=[z_basis,theta_basis,r_basis])
    Br = dist.Field(name='Br', bases=[z_basis,theta_basis,r_basis])
    Ba = dist.Field(name='Ba', bases=[z_basis,theta_basis,r_basis])
    Bz = dist.Field(name='Bz', bases=[z_basis,theta_basis,r_basis])
    Er = dist.Field(name='Er', bases=[z_basis,theta_basis,r_basis])
    Ea = dist.Field(name='Ea', bases=[z_basis,theta_basis,r_basis])
    Ez = dist.Field(name='Ez', bases=[z_basis,theta_basis,r_basis])

    tau_1 = dist.Field(name='tau_1', bases=[z_basis,theta_basis])
    tau_2 = dist.Field(name='tau_2', bases=[z_basis,theta_basis])
    tau_3 = dist.Field(name='tau_3', bases=[z_basis,theta_basis])
    tau_4 = dist.Field(name='tau_4', bases=[z_basis,theta_basis])

    z, th, r = dist.local_grids(z_basis, theta_basis, r_basis)
    rq = dist.Field(name='rq', bases=r_basis)
    rq['g'] = r

    #JOE
    omega_pnsq = dist.Field(name='omega_pnsq', bases=[r_basis])
    omega_pnsq['g'] = 0.5 * (np.tanh((1.528 - r)/0.31) + 1)*(17*ratio/1.5)**2
    # omega_pnsq['g'] = 0.5 * (np.tanh((8.03728 - r)/1.6306) + 1)*(17*ratio/1.5)**2

    envelope_1 = dist.Field(name='envelope_1', bases=[z_basis,theta_basis,r_basis])
    # envelope['g'] = 4*np.exp(-(r-8.03728)**2 / 0.00089888)* np.exp(-(th-0.0)**2 / 0.00089888)*np.exp(-(z-Lz/2)**2 / 0.00089888)
    #envelope_1['g'] = 4*np.exp(-(r-1.528)**2 / 0.00089888)* np.exp(-(th-0.0)**2 /0.00089888)*np.exp(-(z-Lz/4)**2 / 0.00089888)
    envelope_1['g'] = 4*np.exp(-(r-0.6)**2 / 0.00089888)* np.exp(-(th-0.0)**2 /0.00089888)*np.exp(-(z-Lz/4)**2 / 0.00089888)
    #envelope_1['g'] = 4*np.exp(-(r-0.25)**2 / 0.00089888)* np.exp(-(th-0.0)**2 /0.00089888)*np.exp(-(z-Lz/4)**2 / 0.00089888)

    # envelope_2 = dist.Field(name='envelope_2', bases=[z_basis,theta_basis,r_basis])
    # # envelope['g'] = 4*np.exp(-(r-8.03728)**2 / 0.00089888)* np.exp(-(th-0.0)**2 / 0.00089888)*np.exp(-(z-Lz/2)**2 / 0.00089888)
    # envelope_2['g'] = 4*np.exp(-(r-1.528)**2 / 0.00089888)* np.exp(-(th-0.0)**2 /0.00089888)*np.exp(-(z-Lz/4-3.0)**2 / 0.00089888)

    def f(t):
        # return np.exp(-1j*omega_n*t) # simple sinusoidal forcing
        #return np.exp(-1j*omega_n*(t + 0.15*1e6*float(index) )) # simple sinusoidal forcing
        #return np.exp(-1j*omega_n*(t + 0.15*1e6*float(index)*0.25 )) # simple sinusoidal forcing
        return np.exp(-1j*omega_n*(t))*np.exp(-t**2/(1.5**2)) # simple sinusoidal forcing and pause
        

    def forcing(solver):
        return f(solver.sim_time)

    forcing_func = GeneralFunction(dist=dist, domain=domain, layout='g', tensorsig=(), dtype=np.complex128, func=forcing, args=[])
    dr = lambda A: d3.Differentiate(A, coords['r'])
    dz = lambda A: d3.Differentiate(A, coords['z'])
    da = lambda A: d3.Differentiate(A, coords['a'])

    ez, ea, er = coords.unit_vector_fields(dist)
    
    lift_basis = r_basis.derivative_basis(1)
    
    lift = lambda A: d3.Lift(A, lift_basis, -1)

    problem = d3.IVP([V, Br, Ba, Bz, Er, Ea, Ez, tau_1, tau_2, tau_3, tau_4], namespace=locals())
    problem.namespace['omega_n'] = omega_n
    problem.namespace['sigma'] = sigma
    problem.namespace['forcing_func'] = forcing_func

    problem.add_equation("dt(er@V) + Er + sigma*ea@V =  0")
    problem.add_equation("dt(ea@V) + Ea - sigma*er@V =  0")
    problem.add_equation("dt(ez@V) + Ez =  0")
    
    problem.add_equation("rq*dt(Er) - da(Bz) + rq*dz(Ba) = rq*omega_pnsq*er@V + rq*forcing_func*envelope_1")
    problem.add_equation("dt(Ea) + dr(Bz) - dz(Br) + lift(tau_1) = omega_pnsq*ea@V + forcing_func*envelope_1")
    problem.add_equation("rq*dt(Ez) - dr(rq*Ba) + da(Br) + rq*lift(tau_2) = rq*omega_pnsq*ez@V + rq*forcing_func*envelope_1")
    
    problem.add_equation("rq*dt(Br) + da(Ez) - rq*dz(Ea) = 0")
    problem.add_equation("dt(Ba) - dr(Ez) + dz(Er) + lift(tau_3) = 0")
    problem.add_equation("rq*dt(Bz) + dr(rq*Ea) - da(Er) + rq*lift(tau_4) = 0")
    
    problem.add_equation("Ea(r=0.5) = 0")
    problem.add_equation("Ea(r=3.75) = 0")
    problem.add_equation("Ez(r=0.5) = 0")
    problem.add_equation("Ez(r=3.75) = 0")
    
    
    ivp_solver = problem.build_solver('RK222')
    forcing_func.args = [ivp_solver]
    forcing_func.original_args = [ivp_solver]

    ivp_solver.stop_sim_time = 150.0
    # ivp_solver.stop_sim_time = 2500.0
    ivp_solver.stop_wall_time = np.inf
    ivp_solver.stop_iteration = np.inf

    dt = 0.0025
    CFL = flow_tools.CFL(ivp_solver, initial_dt=dt, cadence=5, safety=0.3,
                         max_change=1.5, min_change=0.5, max_dt=0.125, threshold=0.2)
    CFL.add_velocity(V)

    
    shutil.rmtree(output_folder, ignore_errors=True)

    # z_index = np.argmin(np.abs(domain.grid(0) - 26.3/2 - 10.0))

    t1 = time.time()
    analysis_tasks = []
    check = ivp_solver.evaluator.add_file_handler(output_folder, iter=1, max_writes=200000)
    check.add_task(Ez, layout='g', name='Ez')
    check.add_task(Er, layout='g', name='Er')
    check.add_task(Ea, layout='g', name='Ea')
    check.add_task(Bz, layout='g', name='Bz')
    check.add_task(Br, layout='g', name='Br')
    check.add_task(Ba, layout='g', name='Ba')
    # check.add_task(0.5 *rq* (Er*Er+Ea*Ea+Ez*Ez + Br*Br+Ba*Ba + Bz*Bz + omega_pnsq*(V@V)),
    #            name="u_total")
    # S_r = Ea * np.conjugate(Bz) - Ez * np.conjugate(Ba)
    # S_a = Ez * np.conjugate(Br) - Er * np.conjugate(Bz)
    # check.add_task(S_r,layout='g', name="S_r")
    # check.add_task(S_a, layout='g',name="S_a")
    # check.add_task(Ea*np.conjugate(Bz)-Ez*np.conjugate(Ba), name="S_r")
    # check.add_task(Ez*np.conjugate(Br)-Er*np.conjugate(Bz), name="S_a")
    # check.add_task(Er*np.conjugate(Ba)-Ea*np.conjugate(Br), name="S_z")
    #check.add_task((Ea*np.conjugate(Bz)-Ez*np.conjugate(Ba))*np.conjugate(Ea*np.conjugate(Bz)-Ez*np.conjugate(Ba)) + (Ez*np.conjugate(Br)-Er*np.conjugate(Bz))*np.conjugate(Ez*np.conjugate(Br)-Er*np.conjugate(Bz)) + (Er*np.conjugate(Ba)-Ea*np.conjugate(Br))*np.conjugate(Er*np.conjugate(Ba)-Ea*np.conjugate(Br)), name="S^2")
    #check.add_task( ( np.real(Ea)*np.real(Bz)-np.real(Ez)*np.real(Ba) )**2 + ( np.real(Ez)*np.real(Br)-np.real(Er)*np.real(Bz) )**2 +  ( np.real(Er)*np.real(Ba)-np.real(Ea)*np.real(Br) )**2, layout = 'g', name="S^2")
    #analysis_tasks.append(check)

    logger.info(f"Starting simulation for index {index}")
    logger.info("Starting timestepping.")

    while ivp_solver.proceed:
        # if ivp_solver.iteration>=100:
        #   envelope['g'] = 0#4*np.exp(-(r-30.56)**2 / 0.08)* np.exp(-(th-np.pi)**2 / 0.08)*np.exp(-(z-Lz/2)**2 / 0.08)
        ivp_solver.step(dt)
        dt = CFL.compute_timestep()
        logger.info(f"time step {dt}")
        logger.info(f"iteration {ivp_solver.iteration}")
    t2 = time.time()
    logger.info("Elapsed solve time: " + str(t2-t1) + ' seconds')
    logger.info('Iterations: %i' % ivp_solver.iteration)
    logger.info(f"Completed simulation for index {index}")

def main():
    indices = [str(i).zfill(2) for i in range(46,121)]   
    #base_output_path = '/gpfs/home/xxiuhong/scratch/3d_3GHz_S/'
    base_output_path = '/jobtmp/xxiuhong/3d_3GHz_S/'
    
    os.makedirs(base_output_path, exist_ok=True)
    # if len(sys.argv) != 2:
    #     print("Usage: python script.py <index>")
    #     sys.exit(1)
    # index = sys.argv[1]
    # output_folder = f'/gpfs/home/xxiuhong/scratch/3d_3GHz_32_test_{index}/'
    # run_simulation(index, output_folder)
    # shutil.rmtree(base_output_path, ignore_errors=True)
    for index in indices:
        output_folder = f'{base_output_path}3d_3GHz_S_128_{index}/'
        run_simulation(index, output_folder)

if __name__ == "__main__":
    main()
    # MPI.Finalize()
