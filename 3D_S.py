import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

print("Current version of Python is ", sys.version)
import os
import pathlib
import time
import h5py
import numpy as np
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
    Nr = 128
    Na = 128
    Nz = 128
    
    
    ne_min, ne_max = 5e12, 1e13
    num_points = 20
    ne_t = np.linspace(-3, 3, num_points)


    #ne_list = ne_min + (ne_max - ne_min) * 0.5 * (np.tanh(ne_t) + 1)
    ne_list = np.linspace(ne_min, ne_max, num_points)
    idx = int(index)     
    density_for_1e12 = 5.18 * np.sqrt(3)


    density_list = density_for_1e12 * np.sqrt(ne_list / 1e12)
    density = density_list[idx]
    print(density)
    #ratio_list = 0.5*(np.tanh(ne_t)+1)
    #ratio = ratio_list[idx]
    #ratio = np.tanh(0.6*float(index)*0.05)  # density distribution over time, sampling rate 0.05 ms 
    #ratio = 1
    Lz = 20
    #Lz = 40
    #Lz = 400

    sigma= 2.24 # 1.2kG,  3.36 GHz  ratio of cyclotron frequency to plasma frequency
    #sigma= 2.61 # 1.4kG,  3.36 GHz  ratio of cyclotron frequency to plasma frequency
    #sigma= 1.12 # 1.2kG
    #sigma= 0.112 # 1.2kG,  3.36 GHz  ratio of cyclotron frequency to plasma frequency  30GHz
    
    #omega_n = 2/3 # 1 GHz, driving frequency as a fraction of omega_p
    #omega_n = 1 # 1.5 GHz, driving frequency as a fraction of omega_p
    #omega_n = 2/3*0.5 # 0.5 GHz, driving frequency as a fraction of omega_p
    #omega_n = 2/3*2 # 2 GHz, driving frequency as a fraction of omega_p
    omega_n = 2/3*2.5 # 2.5 GHz, driving frequency as a fraction of omega_p
    #omega_n = 1/3*2.5 # 2.5 GHz, driving frequency as a fraction of omega_p
    #omega_n = 2 # 3 GHz, driving frequency as a fraction of omega_p
    #omega_n = 2/3*3.5 # 3.5 GHz, driving frequency as a fraction of omega_p
    #omega_n = 2/3*2.45 # 2.45 GHz, driving frequency as a fraction of omega_p
    
    #density = 5.18*np.sqrt(3) # 1x10e12
    #density = 5.18*3 # 3x10e12, 3x5.18 GHz
    
    #density = 5.18*3*np.sqrt(2*10/3*ratio) # 1x10e13
    #density = 40.15 # 2x10e13
    
    #kz = 41.8 # 2.09 cm^-1
    kz = 31.4 # 2.09 cm^-1 4 cm
    #kz = 15.7 # 2.09 cm^-1 4 cm
    
    coords = d3.CartesianCoordinates('z','a','r')
    
    dist = d3.Distributor(coords, mesh=(8,8),  dtype= np.complex128)
    r_left = 0.1  # 2 cm
    r_right = 1.0 # 20 cm 
    #r_left = 1.0  # 10 cm
    #r_right = 4.0 # 40 cm 
    r_basis = d3.Chebyshev(coords['r'], Nr, bounds=[r_left, r_right], dealias=1)
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

    
    omega_pnsq = dist.Field(name='omega_pnsq', bases=[r_basis])
    
    r1 = 0.25 # 9.5 cm #center of the density
    r0 = 0.25 # -15 cm # center of the antenna
    #r0 = 1.5 # -15 cm # center of the antenna
    #theta0 = np.pi # center of the antenna
    theta0 = 0.0
    w = 0.0475   # 0.95 cm
    epsilon = 0.001 #  <<solution in r direction
    #density distributon
    omega_pnsq['g'] = 0.5 * (np.tanh((r1 - r)/w) + 1)*(density/1.5)**2
    #omega_pnsq['g'] = 0.5 * (np.tanh((r1 - r)/w) + 1)*(density*ratio/1.5)**2
    #omega_pnsq['g'] = (1 - r) * 0.5 * (1 - np.tanh((r - 1) / epsilon))*(density*ratio/1.5)**2
    #omega_pnsq['g'] = (1 - r/2) * 0.5 * (1 - np.tanh((r - 2) / epsilon))*(density*ratio/3)**2
    
    envelope_1 = dist.Field(name='envelope_1', bases=[z_basis,theta_basis,r_basis])
    width = 0.0008988
    #width = 0.0008988*4
    #antenna
    envelope_1['g'] = 4*np.exp(-(r-r0)**2 /width)* np.exp(-(th-theta0)**2 /width)*np.exp(-(z-Lz/4)**2 /width)  # x=-15, y=0
    t_width = 30 # 1 ns

    def f(t):
        return np.exp(-1j*omega_n*(t) + 1j*kz*z )*np.exp(-t**2/(t_width**2)) # simple sinusoidal forcing with fixed kz, pause after 1 ns
        #return np.exp(-1j*omega_n*(t) + 1j*kz*z )

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
    
    problem.add_equation("Ea(r=r_left) = 0")
    problem.add_equation("Ea(r=r_right) = 0")
    problem.add_equation("Ez(r=r_left) = 0")
    problem.add_equation("Ez(r=r_right) = 0")
    
    
    ivp_solver = problem.build_solver('RK222')
    forcing_func.args = [ivp_solver]
    forcing_func.original_args = [ivp_solver]


    ivp_solver.stop_sim_time = 100.0  # 66.6 ns
    ivp_solver.stop_wall_time = np.inf
    ivp_solver.stop_iteration = np.inf

    dt = 0.0025
    CFL = flow_tools.CFL(ivp_solver, initial_dt=dt, cadence=5, safety=0.3,
                         max_change=1.5, min_change=0.5, max_dt=0.125, threshold=0.2)
    CFL.add_velocity(V)

    
    shutil.rmtree(output_folder, ignore_errors=True)


    t1 = time.time()
    analysis_tasks = []
    check = ivp_solver.evaluator.add_file_handler(output_folder, iter=1, max_writes=500)
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
        ivp_solver.step(dt)
        dt = CFL.compute_timestep()
        logger.info(f"time step {dt}")
        logger.info(f"iteration {ivp_solver.iteration}")
    t2 = time.time()
    logger.info("Elapsed solve time: " + str(t2-t1) + ' seconds')
    logger.info('Iterations: %i' % ivp_solver.iteration)
    logger.info(f"Completed simulation for index {index}")
    logger.info(f"Data has been saved at {output_folder}")

def main():
    indices = [str(i).zfill(2) for i in range(0,1)]   
    #base_output_path = '/jobtmp/xxiuhong/3d_14kG_1GHz/'
    #base_output_path = '/jobtmp/xxiuhong/3d_1GHz/'
    #base_output_path = '/jobtmp/xxiuhong/3d_14kG_245GHz_3density/'
    #base_output_path = '/jobtmp/xxiuhong/3d_245GHz_3density/'
    #base_output_path = '/jobtmp/xxiuhong/3d_14kG_245GHz_1density/'
    #base_output_path = '/jobtmp/xxiuhong/3d_14kG_1GHz_3density/'
    #base_output_path = '/jobtmp/xxiuhong/3d_1p2kG_1GHz_1density/'
    #base_output_path = '/jobtmp/xxiuhong/3d_1p2kG_1GHz_3density/'
    #base_output_path = '/jobtmp/xxiuhong/3d_1p2kG_2p45GHz_1density/'
    #base_output_path = '/jobtmp/xxiuhong/3d_1p2kG_2p45GHz_3density/'
    #base_output_path = '/jobtmp/xxiuhong/3d_1p2kG_2p5GHz_2e13_4cm_/'
    base_output_path = '/gpfs/home/xxiuhong/scratch/3d_1p2kG_2p5GHz_2e13_4cm_/'
    
    
    os.makedirs(base_output_path, exist_ok=True)
    for index in indices:
        #output_folder = f'{base_output_path}3d_14kG_1GHz_S_128_{index}/'
        #output_folder = f'{base_output_path}3d_1GHz_S_128_{index}/'
        #output_folder = f'{base_output_path}3d_14kG_245GHz_3density_wave_{index}/'
        #output_folder = f'{base_output_path}3d_245GHz_3density_S_128_{index}/'
        #output_folder = f'{base_output_path}3d_14kG_245GHz_1density_S_128_{index}/'
        #output_folder = f'{base_output_path}3d_14kG_1GHz_3density_wave_{index}/'
        #output_folder = f'{base_output_path}3d_1p2kG_1GHz_1density_wave_{index}/'
        #output_folder = f'{base_output_path}3d_1p2kG_1GHz_3density_wave_{index}/'
        #output_folder = f'{base_output_path}3d_1p2kG_2p45GHz_1density_wave_{index}/'
        #output_folder = f'{base_output_path}3d_1p2kG_2p45GHz_3density_wave_{index}/'
        #output_folder = f'{base_output_path}3d_1p2kG_2p5GHz_1e13_4cm_wave_{index}/'
        output_folder = f'{base_output_path}3d_1p2kG_2p5GHz_2e13_4cm_wave_{index}/'
        
        run_simulation(index, output_folder)

if __name__ == "__main__":
    main()
    # MPI.Finalize()
