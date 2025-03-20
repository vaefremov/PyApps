from typing import Optional, Tuple
import logging
import numpy as np
from scipy.interpolate import interp1d
from scipy.fft import rfft, rfftfreq
from scipy.interpolate import CubicSpline,interp1d,Akima1DInterpolator

from di_lib import di_app
from di_lib.di_app import Context
from di_lib.seismic_cube import DISeismicCube
from di_lib.attribute import DIHorizon3D, DIAttribute2D

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, wait, Future, as_completed

from typing import List
import math
import time

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)

#incr_i = 150
#incr_x = 150
#num_worker = 16
completed_frag = 0
total_frag = 0
num_center_fragments = 3 # Количество центральных фрагментов, который в дальнейшем можно автоматизировать
LOG_INTERVAL = 2 #2 seconds
last_log_time = time.time() 

def move_progress(f: Future):
    global completed_frag, last_log_time
    if f.exception() is None:
        completed_frag += 1
        t = time.time()
        if t - last_log_time >= LOG_INTERVAL:
            LOG.info(f"Completion: {completed_frag*100 // total_frag}")
            job.log_progress("calculation", completed_frag*100 // total_frag)  
            last_log_time = t

def normalizes(a, mode):
    if mode == 'From Bottom':
        norm = (a.max() - a) / (a.max() - a.min())
    else:
        norm = (a - a.min()) / (a.max() - a.min())
    return norm
        
def generate_fragments(min_i, n_i, incr_i, min_x, n_x, incr_x,hdata):
    inc_i = incr_i
    inc_x = incr_x
    res1 = [(i, j, min(n_i-1, i+inc_i-1), min(n_x-1, j+inc_x-1)) for i in range(min_i, n_i-1, inc_i) for j in range(min_x, n_x-1, inc_x)]
    res2 = [(i, j, min(hdata.shape[0]-1, i+inc_i-1), min(hdata.shape[1]-1, j+inc_x-1)) for i in range(0, hdata.shape[0]-1, inc_i) for j in range(0, hdata.shape[1]-1, inc_x)]
    return [(i[0], i[2]-i[0]+1, i[1], i[3]-i[1]+1) for i in res1], [(i[0], i[2]-i[0]+1, i[1], i[3]-i[1]+1) for i in res2]

def compute_fragment(z,cube_in,grid_hor1,grid_hor2,z_step,mode,cube_time_new,countdown_min,countdown_max):
    MAXFLOAT = float(np.finfo(np.float32).max) 
    h_new_all = np.full((grid_hor1.shape[0],grid_hor1.shape[1],len(cube_time_new)), np.nan, dtype = np.float32)
    if np.all(np.isnan(grid_hor1)) == True :
        return z, h_new_all
    elif np.all(np.isnan(grid_hor2)) == True if grid_hor2 is not None else False:
        return z, h_new_all
    else:      
        if mode == 'proportional':
            mask_nan = np.isnan(grid_hor1) | np.isnan(grid_hor2)
            
            valid_i, valid_j = np.where(~mask_nan)
            
            grid1_valid = np.round(grid_hor1[valid_i, valid_j])
            grid2_valid = np.round(grid_hor2[valid_i, valid_j])
            
            ind1 = np.argmin(np.abs(cube_time_new[:, None] - grid1_valid), axis=0)
            ind2 = np.argmin(np.abs(cube_time_new[:, None] - grid2_valid), axis=0)

            for idx in range(len(valid_i)):
                traces = np.arange(grid1_valid[idx], grid2_valid[idx] + z_step, z_step, dtype=float)

                slice_len = ind2[idx] - ind1[idx] + 1
                traces = traces[:slice_len]

                if len(traces) > 1:
                    normalized = normalizes(traces, mode)
                else:
                    normalized = np.zeros_like(traces)

                h_new_all[valid_i[idx], valid_j[idx], ind1[idx] : ind1[idx] + len(normalized)] = normalized
    
        elif mode == 'From Top':
            mask_nan = np.isnan(grid_hor1) | np.isnan(grid_hor2)

            valid_i, valid_j = np.where(~mask_nan)

            grid1_valid = np.round(grid_hor1[valid_i, valid_j])
            grid2_valid = np.round(grid_hor2[valid_i, valid_j])

            ind1 = np.argmin(np.abs(cube_time_new[:, None] - grid1_valid), axis=0)
            ind2 = np.argmin(np.abs(cube_time_new[:, None] - grid2_valid), axis=0)

            for idx in range(len(valid_i)):
                traces = np.arange(cube_time_new[ind1[idx]], countdown_max + z_step, z_step, dtype=float)

                index = np.where(traces == cube_time_new[ind2[idx]])[0][0]

                normalized = normalizes(traces, mode)

                normalized = normalized[:index]

                slice_len = ind2[idx] - ind1[idx]
                normalized = normalized[:slice_len]

                h_new_all[valid_i[idx], valid_j[idx], ind1[idx]:ind1[idx] + len(normalized)] = normalized

        elif mode == 'From Bottom':
            mask_nan = np.isnan(grid_hor1) | np.isnan(grid_hor2)

            valid_i, valid_j = np.where(~mask_nan)

            grid1_valid = np.round(grid_hor1[valid_i, valid_j])
            grid2_valid = np.round(grid_hor2[valid_i, valid_j])

            ind1 = np.argmin(np.abs(cube_time_new[:, None] - grid1_valid), axis=0)
            ind2 = np.argmin(np.abs(cube_time_new[:, None] - grid2_valid), axis=0)

            for idx in range(len(valid_i)):
                traces = np.arange(countdown_min, cube_time_new[ind2[idx]] + z_step, z_step, dtype=float)

                index = np.where(traces == cube_time_new[ind1[idx]])[0][0]

                normalized = normalizes(traces, mode)

                normalized = normalized[index:]

                slice_len = ind2[idx] - ind1[idx] + 1
                normalized = normalized[:slice_len]

                h_new_all[valid_i[idx], valid_j[idx], ind1[idx]:ind1[idx] + len(normalized)] = normalized
        
    return z,h_new_all

def compute_slice(cube_in, hor1, hor2,num_worker,mode,top_shift,top_bottom):
    MAXFLOAT = float(np.finfo(np.float32).max) 
    hdata01 = hor1.get_data()
    hdata01 = np.where((hdata01>= 0.1*MAXFLOAT) | (hdata01== np.inf), np.nan, hdata01)
    hdata1 = np.full((cube_in.n_i-cube_in.min_i,cube_in.n_x-cube_in.min_x), np.nan, dtype = np.float32)
    mask = np.mgrid[cube_in.min_i:cube_in.n_i,cube_in.min_x:cube_in.n_x]
    mask1 = np.mgrid[hor1.min_i:hor1.min_i+hor1.n_i,hor1.min_x:hor1.min_x+hor1.n_x]
    loar1 = np.where((mask1[0]>=cube_in.min_i) & (mask1[0]<cube_in.n_i),True,False) & np.where((mask1[1]>=cube_in.min_x) & (mask1[1]<cube_in.n_x),True,False)
    loar1h = np.where((mask[0]>=hor1.min_i) & (mask[0]<hor1.min_i+hor1.n_i),True,False) & np.where((mask[1]>=hor1.min_x) & (mask[1]<hor1.min_x+hor1.n_x),True,False)
    hdata1[loar1h] = hdata01[loar1]
    hdata1 = hdata1 + top_shift
    
    hdata02 = hor2.get_data() if hor2 is not None else None 
    if hdata02 is not None:
        hdata02 = hor2.get_data()
        hdata2 = np.full((cube_in.n_i-cube_in.min_i,cube_in.n_x-cube_in.min_x), np.nan,dtype = np.float32)
        hdata02 = np.where((hdata02>= 0.1*MAXFLOAT) | (hdata02== np.inf), np.nan, hdata02)
        mask2 = np.mgrid[hor2.min_i:hor2.min_i+hor2.n_i,hor2.min_x:hor2.min_x+hor2.n_x]
        loar2 = np.where((mask2[0]>=cube_in.min_i) & (mask2[0]<cube_in.n_i),True,False) & np.where((mask2[1]>=cube_in.min_x) & (mask2[1]<cube_in.n_x),True,False)
        loar2h = np.where((mask[0]>=hor2.min_i) & (mask[0]<hor2.min_i+hor2.n_i),True,False) & np.where((mask[1]>=hor2.min_x) & (mask[1]<hor2.min_x+hor2.n_x),True,False)
        hdata2[loar2h] = hdata02[loar2]
        if np.nanmean(hdata2) <= np.nanmean(hdata1):
            hdata1, hdata2 = hdata2, hdata1
        hdata2 = hdata2 + top_bottom
    
    z_step_ms = cube_in.time_step / 1000
    cube_time = np.arange(cube_in.data_start, cube_in.data_start  + z_step_ms * cube_in.n_samples, z_step_ms)

    index_max = np.argmin(np.abs(cube_time-np.nanmax(np.round(hdata2))))
    index_min = np.argmin(np.abs(cube_time-np.nanmin(np.round(hdata1))))
    cube_time_new = cube_time[index_min:index_max]
    countdown_min = int(np.nanmin(cube_time_new))
    countdown_max = int(np.nanmax(cube_time_new))
    grid_real, grid_not = generate_fragments(cube_in.min_i, cube_in.n_i, incr_i, cube_in.min_x, cube_in.n_x, incr_x, hdata1)
    new_zr_all = np.full((hdata1.shape[0],hdata1.shape[1],len(cube_time_new)), np.nan,dtype = np.float32)
    
    len_cube = len(grid_not)# Количество фрагметов большого куба
    global total_frag
    total_frag = len(grid_not)
    
    with ProcessPoolExecutor(max_workers=num_worker) as executor:
        futures=[]
        for k in range(len(grid_not)):
            grid_hor1 = hdata1[grid_not[k][0]:grid_not[k][0] + grid_not[k][1],grid_not[k][2]:grid_not[k][2] + grid_not[k][3]]
            grid_hor2 = hdata2[grid_not[k][0]:grid_not[k][0] + grid_not[k][1],grid_not[k][2]:grid_not[k][2] + grid_not[k][3]]
            f = executor.submit(compute_fragment,k,cube_in,grid_hor1,grid_hor2,z_step_ms,mode,cube_time_new,countdown_min,countdown_max)
            f.add_done_callback(move_progress)
            futures.append(f)
            LOG.debug(f"Submitted: {k=}")

        # completed_frag = 0
        for f in as_completed(futures):

            try:
                z,h_new_all = f.result()
                
                h_new_all = h_new_all.astype('float32')
                np.nan_to_num(h_new_all, nan=MAXFLOAT, copy=False)
                cube_in.write_fragment(grid_not[z][0], grid_not[z][2], h_new_all)
            except Exception as e:
                LOG.error(f"Exception: {e}")
    
class Zcube(di_app.DiAppSeismic3D):
    def __init__(self) -> None:
        super().__init__(in_name_par="Input Seismic3D Names", 
                out_name_par="New Name", out_names=[])

    def compute(self, f_in_tup: Tuple[np.ndarray], context: Context) -> Tuple:
        raise NotImplementedError("Shouldn't be called in this application!")

if __name__ == "__main__":
    LOG.debug(f"Starting job ExampleHor1")
    tm_start = time.time()
    job = Zcube()
    cube_in = job.open_input_dataset()
    num_worker = job.description["num_worker"]
    top_shift = job.description["top_shift"]
    top_bottom = job.description["top_bottom"]
    global  incr_i, incr_x
    
    incr_i = job.description["chank_size"]
    incr_x = job.description["chank_size"]
    hor_name1 = job.description["Horizon"][0]
    hor_name2 = job.description["Horizon"][1]
    mode = job.description["mode"]
    hor1 = job.session.get_horizon_3d(cube_in.geometry_name, hor_name1)
    hor2 = job.session.get_horizon_3d(cube_in.geometry_name, hor_name2)
    
    compute_slice(cube_in, hor1, hor2,num_worker,mode,top_shift,top_bottom)

    cube_in.save_statistics_for_horizons(hor_name1, hor_name2)
    LOG.info(f"Processing time (s): {time.time() - tm_start}")