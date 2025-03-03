from typing import Optional, Tuple
import logging
import numpy as np

from scipy.interpolate import CubicSpline

from di_lib import di_app
from di_lib.di_app import Context
from di_lib.seismic_cube import DISeismicCube
from di_lib.attribute import DIHorizon3D, DIAttribute2D

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, wait, Future, as_completed
from typing import List
from numba import jit

import time
import os

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)

#incr_i = 100
#incr_x = 100
completed_frag = 0
total_frag = 0
LOG_INTERVAL = 10
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

def generate_fragments(min_i, n_i, incr_i, min_x, n_x, incr_x,hdata):
    inc_i = incr_i
    inc_x = incr_x
    res1 = [(i, j, min(n_i-1, i+inc_i-1), min(n_x-1, j+inc_x-1)) for i in range(min_i, n_i-1, inc_i) for j in range(min_x, n_x-1, inc_x)]
    res2 = [(i, j, min(hdata.shape[0]-1, i+inc_i-1), min(hdata.shape[1]-1, j+inc_x-1)) for i in range(0, hdata.shape[0]-1, inc_i) for j in range(0, hdata.shape[1]-1, inc_x)]
    return [(i[0], i[2]-i[0]+1, i[1], i[3]-i[1]+1) for i in res1], [(i[0], i[2]-i[0]+1, i[1], i[3]-i[1]+1) for i in res2]

def find_fr(y, ind):
    mask_invalid = np.isnan(ind) | np.isnan(y).all(axis=2)
    
    y_out = np.full(ind.shape, np.nan, dtype=y.dtype)

    ind_fixed = np.where(np.isnan(ind), 0, ind).astype(np.int32)

    valid_mask = (~mask_invalid) & (ind_fixed >= 0) & (ind_fixed < y.shape[2])

    y_out[valid_mask] = y[valid_mask, ind_fixed[valid_mask]]
    
    return y_out

def linear_interpolate_traces(y, c_time, ind, gr_hor):
    y_out = np.full(gr_hor.shape, np.nan)
        
    valid_gr_hor = ~np.isnan(gr_hor)

    valid_y = np.isfinite(y)

    ind_int = np.round(ind)

    valid_idx = (ind_int >= 0) & (ind_int < y.shape[2]) & valid_gr_hor & valid_y.any(axis=2)

    valid_i, valid_j = np.where(valid_idx)

    interp_values = np.array([np.interp(gr_hor[i, j], c_time, y[i, j, :], left=np.nan, right=np.nan)for i, j in zip(valid_i, valid_j)])

    y_out[valid_i, valid_j] = interp_values

    return y_out
@jit
def cubic_interpolate_traces_fast_numba(y, c_time, ind, gr_hor):
    y_out = np.full(gr_hor.shape, np.nan)
        
    valid_gr_hor = ~np.isnan(gr_hor)

    valid_y = np.isfinite(y)

    ind_int = np.round(ind)

    valid_idx = (ind_int >= 0) & (ind_int < y.shape[2]) & valid_gr_hor & valid_y.any(axis=2)

    valid_i, valid_j = np.where(valid_idx)

    return valid_i, valid_j, y_out

def cubic_interpolate_traces(y, c_time, ind1, gr_hor1):
    valid_i, valid_j, y_out = cubic_interpolate_traces_fast_numba(y, c_time, ind1, gr_hor1)

    for i, j in zip(valid_i, valid_j):
        cs = CubicSpline(c_time, y[i, j, :], extrapolate=False)
        y_out[i, j] = cs(gr_hor1[i, j])

    return y_out

def compute_fragment(z,cube_in,grid_hor,cube_time,grid_real,type_interpolation):
    MAXFLOAT = float(np.finfo(np.float32).max) 
    frag_result = np.full((grid_hor.shape[0],grid_hor.shape[1]), np.nan, dtype = np.float32)
    if np.all(np.isnan(grid_hor)) == True :
        return z,frag_result
    else:   
        index_max = np.argmin(np.abs(cube_time-np.nanmax(np.round(grid_hor))))
        index_min = np.argmin(np.abs(cube_time-np.nanmin(np.round(grid_hor))))
        cube_time_new = cube_time[index_min-3:index_max+3]

        fr = cube_in.get_fragment_z(grid_real[z][0],grid_real[z][1], grid_real[z][2],grid_real[z][3],index_min-3,((index_max+3)-(index_min-3)))
        fr = np.where((fr>= 0.1*MAXFLOAT) | (fr== np.inf), np.nan, fr)
        ind = np.round((grid_hor-cube_time_new[0])/(cube_time_new[1] - cube_time_new[0])) #нахождение индекса ближайшего значения
        
        if type_interpolation == "no interpolation":
            frag_result = find_fr(fr, ind) # поиск значений в кубе
        if type_interpolation == "linear":
            frag_result = linear_interpolate_traces(fr, cube_time_new, ind, grid_hor)
        if type_interpolation == "cubic spline":
            frag_result = cubic_interpolate_traces(fr, cube_time_new, ind, grid_hor)
    
    return z,frag_result

def compute_slice(cube_in, hor1,hor2, type_interpolation, shift, distance_between, num_worker,incr_i,incr_x):
    MAXFLOAT = float(np.finfo(np.float32).max) 
    hdata01 = hor1.get_data()
    hdata01 = np.where((hdata01>= 0.1*MAXFLOAT) | (hdata01== np.inf), np.nan, hdata01)
    hdata1 = np.full((cube_in.n_i-cube_in.min_i,cube_in.n_x-cube_in.min_x), np.nan, dtype = np.float32)
    mask = np.mgrid[cube_in.min_i:cube_in.n_i,cube_in.min_x:cube_in.n_x]
    mask1 = np.mgrid[hor1.min_i:hor1.min_i+hor1.n_i,hor1.min_x:hor1.min_x+hor1.n_x]
    loar1 = np.where((mask1[0]>=cube_in.min_i) & (mask1[0]<cube_in.n_i),True,False) & np.where((mask1[1]>=cube_in.min_x) & (mask1[1]<cube_in.n_x),True,False)
    loar1h = np.where((mask[0]>=hor1.min_i) & (mask[0]<hor1.min_i+hor1.n_i),True,False) & np.where((mask[1]>=hor1.min_x) & (mask[1]<hor1.min_x+hor1.n_x),True,False)
    hdata1[loar1h] = hdata01[loar1]
    
    hdata02 = hor2.get_data() if hor2 is not None else None 
    if hdata02 is not None:
        const_step = distance_between/100
        hdata2 = np.full((cube_in.n_i-cube_in.min_i,cube_in.n_x-cube_in.min_x), np.nan,dtype = np.float32)
        hdata02 = np.where((hdata02>= 0.1*MAXFLOAT) | (hdata02== np.inf), np.nan, hdata02)
        mask2 = np.mgrid[hor2.min_i:hor2.min_i+hor2.n_i,hor2.min_x:hor2.min_x+hor2.n_x]
        loar2 = np.where((mask2[0]>=cube_in.min_i) & (mask2[0]<cube_in.n_i),True,False) & np.where((mask2[1]>=cube_in.min_x) & (mask2[1]<cube_in.n_x),True,False)
        loar2h = np.where((mask[0]>=hor2.min_i) & (mask[0]<hor2.min_i+hor2.n_i),True,False) & np.where((mask[1]>=hor2.min_x) & (mask[1]<hor2.min_x+hor2.n_x),True,False)
        hdata2[loar2h] = hdata02[loar2]
        if np.nanmean(hdata2) <= np.nanmean(hdata1):
            hdata1, hdata2 = hdata2, hdata1
        hor_slice = ((hdata2-hdata1) * const_step) + hdata1
    else:
        hor_slice = hdata1 + shift
    
    z_step_ms = cube_in.time_step / 1000
    
    cube_time = np.arange(cube_in.data_start, cube_in.data_start  + z_step_ms * cube_in.n_samples, z_step_ms)
    grid_real, grid_local = generate_fragments(cube_in.min_i, cube_in.n_i, incr_i, cube_in.min_x, cube_in.n_x, incr_x, hor_slice) # на выходе реальные и локальные координаты фрагментов куба в зависимости от размера шага 
    result = np.full((hdata1.shape[0],hdata1.shape[1]), np.nan,dtype = np.float32)

    global total_frag
    total_frag = len(grid_real)
    
    with ProcessPoolExecutor(max_workers=num_worker) as executor:
        futures=[]
        for k in range(len(grid_real)):
            grid_hor = hor_slice[grid_local[k][0]:grid_local[k][0] + grid_local[k][1],grid_local[k][2]:grid_local[k][2] + grid_local[k][3]]
            f = executor.submit(compute_fragment,k,cube_in,grid_hor,cube_time,grid_real,type_interpolation)
            f.add_done_callback(move_progress)
            futures.append(f)
            #LOG.debug(f"Submitted: {k=}")

        for f in as_completed(futures):

            try:
                z,frag_result = f.result()
                
                #LOG.debug(f"Returned {z=}")
                result[grid_local[z][0]:grid_local[z][0] + grid_local[z][1], grid_local[z][2]:grid_local[z][2] + grid_local[z][3]] = frag_result
            except Exception as e:
                LOG.error(f"Exception: {e}")       
        
    result = result.astype('float32')
    np.nan_to_num(result, nan=MAXFLOAT, copy=False)
    if hor_slice.dtype==np.float64:
        hor_slice = hor_slice.astype('float32')
    np.nan_to_num(hor_slice, nan=MAXFLOAT, copy=False)
    return np.vstack([hor_slice[None, :, :]] + [result[None, :, :]])
    
class cubeHorizontsCalculation(di_app.DiAppSeismic3D):
    def __init__(self) -> None:
        super().__init__(in_name_par="Input Seismic3D Names", 
                out_name_par="New Name", out_names=[])

    def compute(self, f_in_tup: Tuple[np.ndarray], context: Context) -> Tuple:
        raise NotImplementedError("Shouldn't be called in this application!")

if __name__ == "__main__":
    LOG.info(f"Starting job ExampleHor1 (pid {os.getpid()})")
    tm_start = time.time()
    job = cubeHorizontsCalculation()
    job.report()
    attr_name = "Amplitude"
    shift = job.description["shift"]
    distance_between = job.description["distance_between"]
    num_worker = job.description["num_worker"]
    incr_i = job.description["chank_size"]
    incr_x = job.description["chank_size"]
    cube_in = job.open_input_dataset()
    hor_name1 = job.description["Horizon"][0]
    hor_name2 = job.description["Horizon"][1] if len(job.description["Horizon"])>1 else None
    type_interpolation = job.description["interpolation"]
    hor1 = job.session.get_horizon_3d(cube_in.geometry_name, hor_name1)
    hor2 = job.session.get_horizon_3d(cube_in.geometry_name, hor_name2) if hor_name2 is not None else None
    dt = compute_slice(cube_in, hor1,hor2, type_interpolation, shift, distance_between, num_worker,incr_i,incr_x)
    f_out = job.session.create_attribute_2d_writer_for_cube(cube_in, job.description["New Name"], attr_name)
    f_out.write_horizon_data(dt[0]) # Not needed if the horizon data have been copied by create_attr (copy_horizon_data=True)
    f_out.write_data(dt[1])

    LOG.info(f"Processing time (s): {time.time() - tm_start}")