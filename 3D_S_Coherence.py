# Import necessary libraries
from mpi4py import MPI
import h5py
import numpy as np
import os
import time
import sys
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import LogFormatter

def process_and_extract_data(folder_number):
    #filename = f"/gpfs/home/xxiuhong/data/xxiuhong/3d_3GHz_S_Slice/3d_3GHz_S_64128_Slices_{folder_number}.h5"
    #filename = f"/gpfs/home/xxiuhong/scratch/3d_3GHz_S_real_Slice/3d_3GHz_S_64128_Slices_{folder_number}.h5"
    #filename = f"/gpfs/home/xxiuhong/scratch/3d_2GHz_S_real_Slice/3d_2GHz_S_64128_Slices_{folder_number}.h5"
    #filename = f"/gpfs/home/xxiuhong/scratch/3d_12kG_1GH_1density/3d_1GH_1density_{folder_number}.h5"
    #filename = f"/gpfs/home/xxiuhong/scratch/3d_14kG_1GHz_1density/3d_14kG_1GHz_1density_{folder_number}.h5"
    #filename = f"/gpfs/home/xxiuhong/scratch/3d_12kG_245GH_3density/3d_245GH_3density_{folder_number}.h5"
    #filename = f"/gpfs/home/xxiuhong/scratch/3d_14kG_245GHz_3density/3d_14kG_245GHz_3density_{folder_number}.h5"
    #filename = "/gpfs/home/xxiuhong/scratch/1p2kG/3d_1p2kG_2p5GHz_1density_S_COV.h5"
    filename = f"/gpfs/home/xxiuhong/scratch/1p2kG_al/3d_1p2kG_2p5GHz_2e13_4cm_wave_{folder_number}.h5"
    print(f"Attempting to open file: {filename}")
    os.utime(filename, (time.time(), time.time()))
    try:
        with h5py.File(filename, 'r') as file:
            print(list(file.keys()))
            Ea1_array = None
            Er1_array = None
            Ez1_array = None
            Ba1_array = None
            Br1_array = None
            Bz1_array = None
            
            S_COV = []
            
            """
            Ea2_array = None
            Er2_array = None
            Ez2_array = None
            Ba2_array = None
            Br2_array = None
            Bz2_array = None
            """
            
            
            
            
            S_array = None
            S_weighted_sum = None
            t_array = []
            z_array = []
            theta_array = []
            rad_array = []
            
            
            
            """
            z_number = 11 #2cm
            
            Er1_array = np.array(file[f'Er_{z_number}'][:])
            Ea1_array = np.array(file[f'Ea_{z_number}'][:])
            Ez1_array = np.array(file[f'Ez_{z_number}'][:])
            Br1_array = np.array(file[f'Br_{z_number}'][:])
            Ba1_array = np.array(file[f'Ba_{z_number}'][:])
            Bz1_array = np.array(file[f'Bz_{z_number}'][:])
            
            z_number = 12 #5cm
            
            Er2_array = np.array(file[f'Er_{z_number}'][:])
            Ea2_array = np.array(file[f'Ea_{z_number}'][:])
            Ez2_array = np.array(file[f'Ez_{z_number}'][:])
            Br2_array = np.array(file[f'Br_{z_number}'][:])
            Ba2_array = np.array(file[f'Ba_{z_number}'][:])
            Bz2_array = np.array(file[f'Bz_{z_number}'][:])
            """
            
            
            
            
            Er1_array = np.array(file[f'Er_data'][:])
            Ea1_array = np.array(file[f'Ea_data'][:])
            Ez1_array = np.array(file[f'Ez_data'][:])
            Br1_array = np.array(file[f'Br_data'][:])
            Ba1_array = np.array(file[f'Ba_data'][:])
            Bz1_array = np.array(file[f'Bz_data'][:])
            """
            # Time
            t_array = np.array(file[f'Time_{folder_number}'][:])
            rad_array = np.array(file[f'r_{folder_number}'][:])
            theta_array = np.array(file[f'theta_{folder_number}'][:])
            """
            t_array = np.array(file[f'time'][:])
            rad_array = np.array(file[f'rad'][:])
            theta_array = np.array(file[f'theta'][:])
            
            #calculating r*
            
            E_total = (Er1_array**2 + Ea1_array**2 + Ez1_array**2 +
            Br1_array**2 + Ba1_array**2 + Bz1_array**2)
            E_total = np.mean(E_total, axis=0)  # shape: (Nr, Nθ)

            
            r_grid, theta_grid = np.meshgrid(rad_array*20, theta_array, indexing='ij')  # shape: (Nr, Nθ)
            
            dr = np.gradient(rad_array)[:, np.newaxis]  # shape: (Nr, 1)
            dtheta = np.gradient(theta_array)[np.newaxis, :]  # shape: (1, Nθ)
            
            dA = r_grid * dr * dtheta  
            
       
            numerator = np.sum(E_total * r_grid * dA)
            numerator = np.real(numerator)
            
            denominator = np.sum(E_total * dA)
            denominator = np.real(denominator)
            r_star = numerator / denominator
            
            
            
            # Define two slices in the z-direction
            total_time = len(t_array)
            
            S_array = (Ea1_array*np.conjugate(Bz1_array)-Ez1_array*np.conjugate(Ba1_array))*np.conjugate(Ea1_array*np.conjugate(Bz1_array)-Ez1_array*np.conjugate(Ba1_array)) + (Ez1_array*np.conjugate(Br1_array)-Er1_array*np.conjugate(Bz1_array))*np.conjugate(Ez1_array*np.conjugate(Br1_array)-Er1_array*np.conjugate(Bz1_array)) + (Er1_array*np.conjugate(Ba1_array)-Ea1_array*np.conjugate(Br1_array))*np.conjugate(Er1_array*np.conjugate(Ba1_array)-Ea1_array*np.conjugate(Br1_array))
            
            S_array = np.real(S_array)
            S_array = np.transpose(S_array,(0,2,1))
            S_COV= np.array(S_array)
            
            int0 = np.argmin(np.abs(t_array - 1.65))
            delta_t = np.diff(t_array)
            delta_t = np.append(delta_t, delta_t[-1])  
            delta_t = delta_t[:, np.newaxis, np.newaxis]  
        
            # sum
            S_weighted_sum = np.sum(S_array[int0:, :, :] * delta_t[int0:], axis=0)

            # average
            total_time = t_array[-1] - t_array[int0]
            S_weighted_average = S_weighted_sum / total_time
        
            
            indr = np.argmin(np.abs(rad_array - 0.75))
            inda = np.argmin(np.abs(theta_array - np.pi))
            
            Ez_slice_1 = np.real(np.array(Ez1_array))
            Er_slice_1 = np.real(np.array(Er1_array))
            Ea_slice_1 = np.real(np.array(Ea1_array))
            """
            
            Ez_slice_2 = np.real(np.array(Ez2_array[:,inda,indr]))
            Er_slice_2 = np.real(np.array(Er2_array[:,inda,indr]))
            Ea_slice_2 = np.real(np.array(Ea2_array[:,inda,indr]))
            """
            
            #E_1 = np.real(np.sqrt((Ez_slice_1**2 + Er_slice_1**2 + Ea_slice_1**2)))
            E_1 = np.zeros((Er_slice_1.shape[0], Er_slice_1.shape[1], Er_slice_1.shape[2]))
            for i in range(Er_slice_1.shape[1]): 
                E_1[:,i,:] = Er_slice_1[:,i,:] * np.sin(theta_array[i]) + Ea_slice_1[:,i,:] * np.cos(theta_array[i])  #Ey
            #E_1[:,i,:] = Er_slice_1[:,i,:] * np.cos(theta_array[i]) - Ea_slice_1[:,i,:] * np.sin(theta_array[i])  #Ex
            #E_1 = np.real(Ez_slice_1)
            E_1 = np.real(E_1)
            E_1 = np.transpose(E_1, (0,2,1))

            """
            
            #E_2 = np.real(np.sqrt(Ez_slice_2**2 + Er_slice_2**2 + Ea_slice_2**2))
            E_2 = None
            E_2 = Er_slice_2 * np.sin(theta_array[inda]) + Ea_slice_2 * np.cos(theta_array[inda]) #Ey
            #E_2 = Er_slice_2 * np.cos(theta_array[inda]) - Ea_slice_2 * np.sin(theta_array[inda]) #Ex
            #E_2 = np.real(Ez_slice_2)
            E_2 = np.real(E_2)
            
            COV = np.zeros((E_1.shape[1], E_1.shape[2]))

            # for t in range(start_step, end_step):
            for i in range(E_1.shape[1]):
                for j in range(E_1.shape[2]):
                    cov_values = []
                    segment_x = E_1[:, i, j]
                    segment_y = E_2
                    if segment_x.size > 1 and np.std(segment_x) > 0 and np.std(segment_y) > 0:
                        cov_value = np.corrcoef(segment_x, segment_y)[0, 1]
                        COV[i, j] = cov_value  
            """
            COV = np.zeros((E_1.shape[1], E_1.shape[2]))
            
            return S_weighted_average,rad_array,theta_array,S_COV,r_star
            
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

def animation(S_arrays,rad_array,theta_array,COV_arrays,r_star):
    
    S_arrays = np.array(S_arrays)
    COV_list = []
    
    ne_min, ne_max = 5e12, 1e13
    num_points = 20
    #ne_t = np.linspace(-3, 3, num_points)
    ne_list = np.linspace(ne_min, ne_max, num_points)
    S_COV1 = []
    S_COV2 = []
    
    indr = np.argmin(np.abs(rad_array - 0.25))
    inda = np.argmin(np.abs(theta_array - 0))
    
    #S_COV1 = np.real(COV_arrays[:,:,inda,indr])
    #S_COV2 = np.real(COV_arrays)
    #print("S_COV1",S_COV1)
    #print("S_COV2",S_COV2)
            
    #COV = np.zeros((S_COV2.shape[2], S_COV2.shape[3]))

    # for t in range(start_step, end_step):
    
    #calculating correlation
    for i in range(len(COV_arrays) - 1):
        #indr = np.argmin(np.abs(rad_array - r_star[i]/20))
        S_COV2 = np.real(COV_arrays[i])      # shape (M_i, J, K)
        S_COV1 = np.real(COV_arrays[i+1][:, indr, inda])  # shape (M_{i+1},)
        
        J, K = S_COV2.shape[1], S_COV2.shape[2]
        COV = np.zeros((J, K))

        for j in range(J):
            for k in range(K):
                cov_values = []
                segment_x = S_COV2[:,j,k]
                segment_y = S_COV1
                m = min(len(segment_x), len(segment_y))
                if m > 1:   
                    x = segment_x[:m]
                    y = segment_y[:m]
                    if np.std(x) > 0 and np.std(y) > 0:
                        cov_value = np.corrcoef(x, y)[0, 1]
                        COV[j, k] = cov_value
        COV_list.append(COV.copy())

    COV_arrays = np.stack(COV_list, axis=0)
    
    #S_arrays = np.log1p(S_arrays) 
    #S_arrays = [np.log(S ) for S in S_arrays]
    
    print("S_arrays shape:", S_arrays.shape)
    print("S_arrays contents:", S_arrays)
    S_arrays = np.sqrt(S_arrays)
    S_arrays = S_arrays + 1e-10
    #S_arrays = np.log1p(S_arrays)
    #S_arrays = S_arrays + 1e-10
    #S_arrays = np.log1p(S_arrays)
    S_max = np.max([np.max(S) for S in S_arrays])
    S_min = np.min([np.min(S) for S in S_arrays])
    print(f"S_max: {S_max}, S_min: {S_min}")
    #S_arrays = (S_arrays-S_min)/(S_max-S_min)
    #S_max = np.max([np.max(S) for S in S_arrays])
    #S_min = np.min([np.min(S) for S in S_arrays])
    #S_arrays = (S_arrays-S_min)/(S_max-S_min)
    
    #norm = Normalize(vmin=S_min, vmax=S_max)
    norm = LogNorm(vmin=S_min, vmax=S_max)
    rad_array = rad_array*20
    #r_star_list = r_star*20
    
    S = np.abs(S_arrays[0])  
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}) #polarised 
    fig.subplots_adjust(top=0.85)  
    quad = ax.pcolormesh(theta_array, rad_array, S, shading='auto', norm=norm, cmap="coolwarm")
    #cb = plt.colorbar(quad, ax=ax)
    cb = plt.colorbar(quad, ax=ax, format=LogFormatter(10, labelOnlyBase=False))
    cb.set_label('S (log scale)')  

    
    title1 = ax.set_title("")

    def update(frame):
        
        S = np.abs(S_arrays[frame])
        
        #np.save(f"/gpfs/home/xxiuhong/scratch/1p2kG/frames_data/frame_{frame:04d}.npy", S)
        
        quad = ax.pcolormesh(theta_array, rad_array, S, shading='auto', norm=norm, cmap="coolwarm")
        ne_min, ne_max = 5e12, 1e13
        num_points = 20
        ne_t = np.linspace(-3, 3, num_points)


    
        ne_list = np.linspace(ne_min, ne_max, num_points)
        density_for_1e12 = 5.18 * np.sqrt(3)


        density_list = density_for_1e12 * np.sqrt(ne_list / 1e12)
        density = density_list[frame]
        #density = 5e12 + (2e13-5e12)*0.5*(np.tanh(float(frame)+1))
        
        #time_in_ms = float(frame) * 0.005  
        #time_in_ms = float(frame) * 0.05  
        fixed_title1 = "B=1.2kG,driving frequency=2.5GHz,distance = 10cm"#,density=2x10e13"
        #fixed_title1 = "B=1.4kG,driving frequency=1GHz,density=1x10e12"
        #fixed_title1 = "B=1.2kG,driving frequency=2.45GHz,density=3x10e12"
        #fixed_title1 = "B=1.4kG,driving frequency=2.45GHz,density=3x10e12"
        dynamic_title1= f'n_e = {ne_list[frame]:.2e},r* = {r_star[frame]:.2e}cm'
        #dynamic_title1 = f't = {density:.0e}'
        #dynamic_title1 = f't = {time_in_ms:.3f} ms'
        title1.set_text(f"{fixed_title1}\n{dynamic_title1}")
        
        plt.savefig(f"/gpfs/home/xxiuhong/scratch/1p2kG/frames_images/frame_{frame:04d}.png", dpi=150)
        

        return quad,title1

    
    frames = len(S_arrays)
    ani1 = FuncAnimation(fig, update, frames=frames, blit=False)
    
    global_max = max(np.max(COV) for COV in COV_arrays)
    global_min = min(np.min(COV) for COV in COV_arrays)
    print(global_max)
    print(global_min)
    
    
    fig2, ax2 = plt.subplots(subplot_kw={'projection': 'polar'})
    fig2.subplots_adjust(top=0.85)  
    norm2 = Normalize(vmin=-1, vmax=1)
    #norm = LogNorm(vmin=global_min, vmax=global_max)
    c2 = ax2.pcolormesh(theta_array, rad_array, COV_arrays[0],shading='auto', norm=norm2,cmap="coolwarm")
    #cb = plt.colorbar(c, ax=ax, format=LogFormatter(10, labelOnlyBase=False))
    cb2 = plt.colorbar(c2, ax=ax2)
    #cb.set_label('log(S)') 
    cb2.set_label('Cross-correlation Ey-Ey') 
    #cb.set_label('Cross-correlation Ex-Ex') 
    #cb.set_label('Cross-correlation E-E')
    title2 = ax2.set_title('')
    
    def update2(frame):
        COV = COV_arrays[frame]
        print(np.shape(COV))
        print(COV_arrays)
        
        ne_min, ne_max = 5e12, 1e13
        num_points = 20
        ne_t = np.linspace(-3, 3, num_points)


    
        ne_list = np.linspace(ne_min, ne_max, num_points)
        density_for_1e12 = 5.18 * np.sqrt(3)


        density_list = density_for_1e12 * np.sqrt(ne_list / 1e12)
        density = density_list[frame]
        
        #time_in_ms = float(frame) * 0.005
        time_in_ms = float(frame) * 0.05 
        #density = 5e12 + (2e13-5e12)*0.5*(np.tanh(float(frame)+1)
        density = 5e12 + (2e13-5e12)*0.5*(np.tanh(float(frame)+1))
        c2 = ax2.pcolormesh(theta_array, rad_array, COV, shading='auto', norm=norm2, cmap="coolwarm")
        c2.set_array(COV.flatten())
        #fixed_title2 = "B=1.2kG,driving frequency=1GHz"
        #fixed_title2 = "B=1.2kG,driving frequency=1GHz,density=1x10e12"
        #fixed_title2 = "B=1.4kG,driving frequency=1GHz,density=1x10e12"
        #fixed_title2 = "B=1.2kG,driving frequency=2.45GHz,density=3x10e12"
        #fixed_title2 = "B=1.4kG,driving frequency=2.45GHz,density=3x10e12"
        fixed_title2 = "B=1.2kG,driving frequency=2.5GHz,distance = 10cm"
        #dynamic_title2 = f't = {time_in_ms:.3f} ms'
        dynamic_title2 = f'density = {density:.0e} '
        title2.set_text(f"{fixed_title2}\n{dynamic_title2}")
        print(np.max(COV))
        print(np.min(COV))
    
        return c2,title2
        
    frames = len(COV_arrays)-10
    ani2 = FuncAnimation(fig2, update2, frames=frames, blit=False)
    
    
    
    return ani1,ani2
    #return ani1


    
    
def main():
    
    #ani1_path ='/gpfs/home/xxiuhong/scratch/3d_12kG_1GH_1density/wave/3d_12kG_1GH_1density_S_1.mp4'
    #ani1_path ='/gpfs/home/xxiuhong/scratch/3d_14kG_1GHz_1density/wave/3d_14kG_1GHz_1density_S_1.mp4'
    #ani1_path ='/gpfs/home/xxiuhong/scratch/3d_12kG_245GH_3density/wave/3d_12kG_2450MHz_3density_S_1.mp4'
    #ani1_path ='/gpfs/home/xxiuhong/scratch/3d_14kG_245GHz_3density/wave/3d_14kG_2p45GHz_3density_S_1.mp4'
    #ani2_path ='/gpfs/home/xxiuhong/scratch/3d_12kG_1GH_1density/wave/3d_12kG_1GH_1density_COV_1.mp4'
    #ani2_path ='/gpfs/home/xxiuhong/scratch/3d_14kG_1GHz_1density/wave/3d_14kG_1GHz_1density_COV_1.mp4'
    #ani2_path ='/gpfs/home/xxiuhong/scratch/3d_12kG_245GH_3density/wave/3d_12kG_2450MHz_3density_COV_1.mp4'
    #ani2_path ='/gpfs/home/xxiuhong/scratch/3d_14kG_245GHz_3density/wave/3d_14kG_2p45GHz_3density_COV_1.mp4'
    ani1_path = '/gpfs/home/xxiuhong/scratch/1p2kG/3d_1p2kG_2p5GHz_2e13_4cm_wave.mp4'
    ani2_path ='/gpfs/home/xxiuhong/scratch/1p2kG/3d_1p2kG_2p5GHz_2e13_4cm_cor.mp4'
    
    save_dir = "/gpfs/home/xxiuhong/data/xxiuhong/1p2kG/frames_data"
    os.makedirs(save_dir, exist_ok=True)
    index = [str(i).zfill(2) for i in range(0,20)]
    S = []
    COV = []
    r_star = []
    for ind in index:
        
        results = process_and_extract_data(ind)
        if results is not None:
            print(f"Shape of S_array for folder {ind}: {results[0].shape}")
            print(f"Shape of COV for folder {ind}: {results[3].shape}")
            print(f"shape of theta for folder {ind}: {results[2].shape}")
            print(f"value of theta for folder {ind}: {results[2]}")
            r_array = results[1]
            theta_array = results[2]
            S.append(results[0])
            COV.append(results[3])
            r_star.append(results[4])
            frame_idx = int(ind)
            
            np.save(f"/gpfs/home/xxiuhong/scratch/1p2kG/frames_data/frame_{frame_idx:04d}.npy", results[0])
            #np.save(f"/gpfs/home/xxiuhong/data/xxiuhong/1p2kG/frames_data/frame_{frame_idx:04d}.npy", results[0])
    np.save("/gpfs/home/xxiuhong/scratch/1p2kG/rad_array.npy", r_array)
    np.save("/gpfs/home/xxiuhong/scratch/1p2kG/theta_array.npy", theta_array)
    np.save("/gpfs/home/xxiuhong/scratch/1p2kG/r_star.npy", results[4])
    #np.save("/gpfs/home/xxiuhong/data/xxiuhong/1p2kG/rad_array.npy", r_array)
    #np.save("/gpfs/home/xxiuhong/data/xxiuhong/1p2kG/theta_array.npy", theta_array)
            
    #ani1,ani2 = animation(S,r_array,theta_array,COV)
    
    ani1,ani2 = animation(S,r_array,theta_array,COV,r_star)
    writer = FFMpegWriter(fps=1, metadata=dict(artist='Me'), bitrate=5000)
    ani1.save(ani1_path, writer= writer)
    ani2.save(ani2_path, writer= writer)

if __name__ == "__main__":
    main()
