from mpi4py import MPI
import h5py
import numpy as np
import os
import glob
import sys
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.colors import SymLogNorm
from matplotlib.ticker import FuncFormatter
import re
from matplotlib.colors import Normalize

def sort_key(fname):
    # Extract the number after "_s" in the filename
    m = re.search(r"_s(\d+)\.h5$", fname)
    if m:
        return int(m.group(1))
    else:
        # If no match, return the filename itself as a fallback sorting key
        return fname

def scientific_formatter(x, pos):
    return f"{x:.1e}"

def process_and_extract_data(folder_number,folder_name,folder_path,z_number):
    """
    All MPI processes collaboratively process files in the folder:
      1. All processes open the same file simultaneously
      2. Split the time dimension; each process handles part of the data
      3. Gather results from all processes back to rank 0
      4. Rank 0 merges data from all files
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # 1. Rank 0 builds the file list and broadcasts it to all processes
    if rank == 0:
        
        pattern = os.path.join(folder_path, f"{folder_name}_s*.h5")
        file_list = sorted(glob.glob(pattern))
        file_list = sorted(glob.glob(pattern), key=sort_key)
        #file_list = file_list[:5]
        #file_list = file_list[1:-1]

        if not file_list:
            print(f"Rank0: no matching file find: {pattern}")
            sys.exit(1)
        print(f"Rank0: finding matched files: total number {len(file_list)}")
        for fname in file_list:
            print(f"    {fname}")
    else:
        file_list = None
    file_list = comm.bcast(file_list, root=0)

    # Buffers for processed data (only valid on rank 0)
    combined_Ez_list = []
    combined_Bz_list = []
    combined_Er_list = []
    combined_Br_list = []
    combined_Ea_list = []
    combined_Ba_list = []
    combined_t_list = []
    spatial_coords = None  # store spatial coordinates (z, theta, rad)

    # 2. Process each file
    for filename in file_list:
        if rank == 0:
            print(f"Rank0: processing: {filename}")
        # All processes open the same file
        with h5py.File(filename, 'r') as f:
            Ez_dataset = f.get('tasks')['Ez']
            Er_dataset = f.get('tasks')['Er']
            Ea_dataset = f.get('tasks')['Ea']
            Bz_dataset = f.get('tasks')['Bz']
            Br_dataset = f.get('tasks')['Br']
            Ba_dataset = f.get('tasks')['Ba']
            
            # Extract time information (assumed stored in first dimension attribute 'sim_time')
            t = np.array(Ez_dataset.dims[0]['sim_time'], dtype=np.float64)
            total_frames = t.shape[0]
            if rank == 0 :
                print(f"File: {filename}")
                print("Time array:", t)
                print(f"Time min: {t.min()}, max: {t.max()}, length: {t.shape[0]}")
            # Split time frames among processes
            all_indices = np.arange(total_frames)
            indices_split = np.array_split(all_indices, size)
            local_indices = indices_split[rank]
            
            if spatial_coords is None:
                z = np.array(Ez_dataset.dims[1][0])
                theta = np.array(Ez_dataset.dims[2][0])
                rad = np.array(Ez_dataset.dims[3][0])
                spatial_coords = (z, theta, rad)
            
            # Select data at the desired distance z from the antenna
            z_number = np.float64(z_number)
            z_aim = 20 / 4 + z_number/20
            #z_aim = 20 / 4 + 1.5#30
            
            
            indz = np.argmin(np.abs(z - z_aim))
            # Each process reads its assigned time frames
            local_Ez = np.array(Ez_dataset[local_indices, indz,:,:])
            local_Bz = np.array(Bz_dataset[local_indices, indz,:,:])
            local_Er = np.array(Er_dataset[local_indices, indz,:,:])
            local_Br = np.array(Br_dataset[local_indices, indz,:,:])
            local_Ea = np.array(Ea_dataset[local_indices, indz,:,:])
            local_Ba = np.array(Ba_dataset[local_indices, indz,:,:])
            local_t = t[local_indices]
            
            
        # 3.Gather data from all processes to rank 0
        gathered_Ez = comm.gather(local_Ez, root=0)
        gathered_Bz = comm.gather(local_Bz, root=0)
        gathered_Ea = comm.gather(local_Ea, root=0)
        gathered_Ba = comm.gather(local_Ba, root=0)
        gathered_Er = comm.gather(local_Er, root=0)
        gathered_Br = comm.gather(local_Br, root=0)
        gathered_t = comm.gather(local_t, root=0)
        if rank == 0:
            # Concatenate data along the time axis
            file_Ez = np.concatenate(gathered_Ez, axis=0)
            file_Bz = np.concatenate(gathered_Bz, axis=0)
            file_Ea = np.concatenate(gathered_Ea, axis=0)
            file_Ba = np.concatenate(gathered_Ba, axis=0)
            file_Er = np.concatenate(gathered_Er, axis=0)
            file_Br = np.concatenate(gathered_Br, axis=0)
            
            file_t = np.concatenate(gathered_t, axis=0)
            combined_Ez_list.append(file_Ez)
            combined_Bz_list.append(file_Bz)
            combined_Er_list.append(file_Er)
            combined_Br_list.append(file_Br)
            combined_Ea_list.append(file_Ea)
            combined_Ba_list.append(file_Ba)
            combined_t_list.append(file_t)
    
    # 4. Merge all data on rank 0
    if rank == 0:
        Ez_array_combined = np.concatenate(combined_Ez_list, axis=0)
        Bz_array_combined = np.concatenate(combined_Bz_list, axis=0)
        Er_array_combined = np.concatenate(combined_Er_list, axis=0)
        Br_array_combined = np.concatenate(combined_Br_list, axis=0)
        Ea_array_combined = np.concatenate(combined_Ea_list, axis=0)
        Ba_array_combined = np.concatenate(combined_Ba_list, axis=0)
        t_array_combined = np.concatenate(combined_t_list, axis=0)
        z_array, theta_array, rad_array = spatial_coords
        print("Rank0: combined data shape :", Ez_array_combined.shape)
        print("Time array:", t_array_combined)
            
        print(f"Time min: {t_array_combined.min()}, max: {t_array_combined.max()}, length: {t_array_combined.shape[0]}")
        return (t_array_combined, Ez_array_combined,Bz_array_combined,Er_array_combined,Br_array_combined,Ea_array_combined,Ba_array_combined, theta_array, rad_array, z_array)
    else:
        return None

def create_animation(t_array, S, theta_array, rad_array, z_array, folder_number,z_number):
   
    S_selected = np.real(S)
   
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    fig.subplots_adjust(top=0.85)
    
    
    vmin = S_selected.min()
    vmax = S_selected.max()
    linthresh = 1e-4
    norm = SymLogNorm(linthresh=linthresh, vmin=vmin, vmax=vmax, base=10)
    #norm = Normalize(vmin=vmin, vmax=vmax)
    
   
    rad_array_scaled = 20 * rad_array
    
    S_transformed = np.transpose(S_selected, (0, 2, 1))
    
    
    quad = ax.pcolormesh(theta_array, rad_array_scaled, S_transformed[0, :, :],
                           shading='auto', norm=norm, cmap="coolwarm")
    cbar = fig.colorbar(quad, ax=ax)
    #cbar.set_label(S)
    cbar.set_label('log(S)')
    formatter = FuncFormatter(scientific_formatter)
    cbar.ax.yaxis.set_major_formatter(formatter)
    cbar.update_ticks()

    
    frame_indices = np.arange(0, len(t_array), 1)
    title = ax.set_title("")
    #density_val = 5.18*3*np.sqrt(10/3)  
    
    
    def update(frame):
        quad.set_array(S_transformed[frame, :, :].ravel())
        quad.changed()
        cbar.update_normal(quad)
        cbar.ax.yaxis.set_major_formatter(formatter)
        cbar.update_ticks()
        time_in_ms = float(t_array[frame]) * 2 / 3  
        fixed_title = f"B=1.2kG, driving frequency=2.5GHz, peak density=1e13, distance:{z_number}cm"
        dynamic_title = f't = {time_in_ms:.4f} ns'
        title.set_text(f"{fixed_title}\n{dynamic_title}")
        return quad, title

    ani = FuncAnimation(fig, update, frames=frame_indices, blit=False)
    return ani

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    #if len(sys.argv) != 2:
        #if rank == 0:
            #print("Usage: mpiexec -n <num_procs> python3 3D_S_DATA_MPI.py <z_number>")
        #sys.exit(1)
        
    #z_number = sys.argv[1]
    z_number = 10
    
    if len(sys.argv) != 2:
        if rank == 0:
            print("Usage: mpiexec -n <num_procs> python3 3D_S_DATA_MPI.py <serial>")
        sys.exit(1)
    raw_serial = sys.argv[1].strip()
    # force two-digit format
    serial = f"{int(raw_serial):02d}"
    
    folder_number = 00

    #folder_name = f"3d_1p2kG_2p45GHz_1density_wave_{folder_number}"
    #folder_path = f"/jobtmp/xxiuhong/3d_1p2kG_2p45GHz_1density/{folder_name}"
    #output_path = '/gpfs/home/xxiuhong/scratch/1p2kG/3d_1p2kG_2p45GHz_1density_s.mp4'
    #folder_name = f"3d_1p2kG_1GHz_1density_wave_{folder_number}"
    #folder_path = f"/jobtmp/xxiuhong/3d_1p2kG_1GHz_1density/{folder_name}"
    #output_path = '/gpfs/home/xxiuhong/scratch/1p2kG/3d_1p2kG_1GHz_1density_s.mp4'
    """
    folder_name = f"3d_1p2kG_1GHz_3density_wave_{folder_number}"
    folder_path = f"/jobtmp/xxiuhong/3d_1p2kG_1GHz_3density/{folder_name}"
    output_path = '/gpfs/home/xxiuhong/scratch/1p2kG/3d_1p2kG_1GHz_3density_s.mp4'
    output_filename = f"/gpfs/home/xxiuhong/scratch/1p2kG/Ez_data_1p2kG_1GHz_3density_{folder_number}.h5"
    
    """
    """
    folder_name = f"3d_1p2kG_0p5GHz_3density_wave_{folder_number}"
    folder_path = f"/jobtmp/xxiuhong/3d_1p2kG_0p5GHz_3density/{folder_name}"
    output_path = '/gpfs/home/xxiuhong/scratch/1p2kG/3d_1p2kG_0p5GHz_3density_s.mp4'
    output_filename = f"/gpfs/home/xxiuhong/scratch/1p2kG/Ez_data_1p2kG_0p5GHz_3density_{folder_number}.h5"
    
    """
    """
    folder_name = f"3d_1p2kG_2p5GHz_2e13_4cm_wave_19"
    folder_path = f"/gpfs/home/xxiuhong/scratch/3d_1p2kG_2p5GHz_2e13_4cm_/{folder_name}"
    #folder_path = f"/jobtmp/xxiuhong/3d_1p2kG_2p5GHz_2e13_4cm_/{folder_name}"
    #output_path = f'/gpfs/home/xxiuhong/scratch/1p2kG/3d_1p2kG_2p5GHz_1e13_4cm_{z_number}_test4.mp4'
    output_filename = f"/gpfs/home/xxiuhong/scratch/1p2kG_al/3d_1p2kG_2p5GHz_2e13_4cm_wave_19.h5"
    """
    
    
    folder_name = f"3d_1p2kG_2p5GHz_2e13_4cm_wave_{serial}"
    folder_path = f"/gpfs/home/xxiuhong/scratch/3d_1p2kG_2p5GHz_2e13_4cm_/{folder_name}"
    #folder_path = f"/jobtmp/xxiuhong/3d_1p2kG_2p5GHz_2e13_4cm_/{folder_name}"
    #output_path = f'/gpfs/home/xxiuhong/scratch/1p2kG/3d_1p2kG_2p5GHz_1e13_4cm_{z_number}_test4.mp4'
    output_filename = f"/gpfs/home/xxiuhong/scratch/1p2kG_al/3d_1p2kG_2p5GHz_2e13_4cm_wave_{serial}.h5"
    
    result = process_and_extract_data(folder_number,folder_name,folder_path,z_number)
    
    

    if rank == 0:
        t_array, Ez,Bz, Er,Br,Ea,Ba, theta_array, rad_array, z_array = result
        #output_filename = f"/gpfs/home/xxiuhong/scratch/1p2kG/Ez_data_1p2kG_1GHz_1density_{folder_number}.h5"
        #output_filename = f"/gpfs/home/xxiuhong/scratch/1p2kG/Ez_data_1p2kG_2p45GHz_1density_{folder_number}.h5"
        print(f"Saving data to {output_filename} ...")
        
        """
        try:
            with h5py.File(output_filename, 'a') as hf:
        # --- Handle Ez_data ---
        # Check if the new chunked dataset exists; if not, create it.
                if "Ez_data_chunked" not in hf:
                    if "Ez_data" in hf:
                        # Copy the old contiguous data from "Ez_data"
                        old_ez = hf["Ez_data"][:]
                        print("Old Ez_data shape:", old_ez.shape)
                        # Create a new chunked dataset with the old data
                        hf.create_dataset(
                            "Ez_data_chunked",
                            data=old_ez,
                            maxshape=(None,) + old_ez.shape[1:],
                            chunks=True
                        )
                        print("Created 'Ez_data_chunked' from 'Ez_data'.")
                    else:
                        # If neither exists, create a new chunked dataset with new_Ez
                        hf.create_dataset(
                            "Ez_data_chunked",
                            data=Ez,
                            maxshape=(None,) + Ez.shape[1:],
                            chunks=True
                        )
                        print("Created new 'Ez_data_chunked'.")
                # Append new data to "Ez_data_chunked"
                dset_ez = hf["Ez_data_chunked"]
                old_size = dset_ez.shape[0]
                new_size = old_size + Ez.shape[0]
                dset_ez.resize((new_size,) + dset_ez.shape[1:])
                dset_ez[old_size:] = Ez
                print(f"Appended new Ez_data. New shape: {dset_ez.shape}")
        
                # --- Handle time ---
                if "time_chunked" not in hf:
                    if "time" in hf:
                        # Copy the old time data
                        old_time = hf["time"][:]
                        print("Old time shape:", old_time.shape)
                        hf.create_dataset(
                            "time_chunked",
                            data=old_time,
                            maxshape=(None,),
                            chunks=True
                        )
                        print("Created 'time_chunked' from 'time'.")
                    else:
                        hf.create_dataset(
                            "time_chunked",
                            data=t_array,
                            maxshape=(None,),
                            chunks=True
                        )
                        print("Created new 'time_chunked'.")
                # Append new time data
                dset_time = hf["time_chunked"]
                old_size = dset_time.shape[0]
                new_size = old_size + t_array.shape[0]
                dset_time.resize((new_size,))
                dset_time[old_size:] = t_array
                print(f"Appended new time data. New shape: {dset_time.shape}")
                

            print("Data successfully appended!")
        except Exception as e:
            print(f"Failed to append data: {e}")
            sys.exit(1)
        """
    
        try:
            with h5py.File(output_filename, 'w') as hf:
                hf.create_dataset("time", data=t_array)
                hf.create_dataset("Ez_data", data=Ez)
                hf.create_dataset("Bz_data", data=Bz)
                hf.create_dataset("Er_data", data=Er)
                hf.create_dataset("Br_data", data=Br)
                hf.create_dataset("Ea_data", data=Ea)
                hf.create_dataset("Ba_data", data=Ba)
                hf.create_dataset("theta", data=theta_array)
                hf.create_dataset("rad", data=rad_array)
                hf.create_dataset("z", data=z_array)
            print("Data successfully saved!")
        except Exception as e:
            print(f"Failed to save data: {e}")
            sys.exit(1)
        
        """
        #t_array, Ez, theta_array, rad_array, z_array = result
        #print("Animation creating")
        #ani = create_animation(t_array, Ez, theta_array, rad_array, z_array, folder_number,z_number)
        #writer = FFMpegWriter(fps=120, metadata=dict(artist='Me'), bitrate=5000)
        #ani.save(output_path, writer=writer)
        #print("Animation successfully created:", output_path)
        """
if __name__ == "__main__":
    main()
