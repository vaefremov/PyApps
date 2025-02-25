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
#import cProfile
from typing import List

import time
import os

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)

incr_i = 100
incr_x = 100
completed_frag = 0
total_frag = 0

def move_progress(f: Future):
    global completed_frag
    if f.exception() is None:
        completed_frag += 1
        LOG.info(f"Completion: {completed_frag*100 // total_frag}")
        #job.log_progress("calculation", completed_frag*100 // total_frag)  

def generate_fragments(min_i, n_i, incr_i, min_x, n_x, incr_x,hdata):
    inc_i = incr_i
    inc_x = incr_x
    res1 = [(i, j, min(n_i-1, i+inc_i-1), min(n_x-1, j+inc_x-1)) for i in range(min_i, n_i-1, inc_i) for j in range(min_x, n_x-1, inc_x)]
    res2 = [(i, j, min(hdata.shape[0]-1, i+inc_i-1), min(hdata.shape[1]-1, j+inc_x-1)) for i in range(0, hdata.shape[0]-1, inc_i) for j in range(0, hdata.shape[1]-1, inc_x)]
    return [(i[0], i[2]-i[0]+1, i[1], i[3]-i[1]+1) for i in res1], [(i[0], i[2]-i[0]+1, i[1], i[3]-i[1]+1) for i in res2]

def cut_intervals(y, ind1):
    mask_invalid = np.isnan(ind1) | np.isnan(y).all(axis=2)
    
    y_out = np.full(ind1.shape, np.nan, dtype=y.dtype)

    ind1_fixed = np.where(np.isnan(ind1), 0, ind1).astype(np.int32)

    valid_mask = (~mask_invalid) & (ind1_fixed >= 0) & (ind1_fixed < y.shape[2])

    y_out[valid_mask] = y[valid_mask, ind1_fixed[valid_mask]]
    return y_out

def linear_interpolate_traces(y, c_time, ind1, gr_hor1):
    y_out  = []               
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            if np.isnan(gr_hor1[i,j]) or np.isnan(y[i,j,:]).all():
                y_out.append([np.nan])
                continue
            else:
                ind_1 = int(ind1[i,j])
                if np.isnan(y[i,j,ind_1]):
                    y_out.append([np.nan])
                    continue
                else:
                    good_idx = np.where( np.isfinite(y[i,j,:]) )
                    x0 = gr_hor1[i,j]
                    try :
                        y_out.append(np.interp(x0, c_time[good_idx], y[i,j,good_idx][0,:], left = np.nan, right = np.nan ))
                    except:
                        y_out.append([np.nan])
                        continue
    return y_out

def cubic_interpolate_traces(y, c_time, ind1, gr_hor1):
    y_out = []
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            if np.isnan(gr_hor1[i,j]) or np.isnan(y[i,j,:]).all():
                y_out.append([np.nan])
                continue
            else:
                ind_1 = int(ind1[i,j])
                if np.isnan(y[i,j,ind_1]):
                    y_out.append([np.nan])
                    continue
                else:
                    good_idx = np.where( np.isfinite(y[i,j,:]) )
                    x0 = gr_hor1[i,j]
                    try :
                        y_out.append(CubicSpline(c_time[good_idx], y[i,j,good_idx][0,:], extrapolate=False )(x0))
                    except:
                        y_out.append([np.nan])
                        continue
    return y_out

def compute_fragment(z,cube_in,grid_hor,cube_time,grid_real,type_interpolation):
    MAXFLOAT = float(np.finfo(np.float32).max) 
    h_new_all = np.full((grid_hor.shape[0],grid_hor.shape[1]), np.nan, dtype = np.float32)
    #if np.all(np.isnan(grid_hor)) == True :
    #    return z,h_new_all
    #else:   
        #index_max = np.argmin(np.abs(cube_time-np.nanmax(np.round(grid_hor))))
        #index_min = np.argmin(np.abs(cube_time-np.nanmin(np.round(grid_hor))))
        #cube_time_new = cube_time[index_min-3:index_max+3]

        #fr = cube_in.get_fragment_z(grid_real[z][0],grid_real[z][1], grid_real[z][2],grid_real[z][3],index_min-3,((index_max+3)-(index_min-3)))
        #fr = np.where((fr>= 0.1*MAXFLOAT) | (fr== np.inf), np.nan, fr)
        #indxs1 = np.round((grid_hor-cube_time_new[0])/(cube_time_new[1] - cube_time_new[0]))
        
        #if type_interpolation == "no interpolation":
            #h_new_all = cut_intervals(fr, indxs1)
            #h_new_all = np.full((grid_hor.shape[0],grid_hor.shape[1]), np.nan, dtype = np.float32)
        #if type_interpolation == "linear":
        #    fr_intv = linear_interpolate_traces(fr, cube_time_new, indxs1, grid_hor)
        #if type_interpolation == "cubic spline":
        #    fr_intv = cubic_interpolate_traces(fr, cube_time_new, indxs1, grid_hor)
        #if type_interpolation != "no interpolation": #### Временно
        #    h_new_all = np.full((grid_hor.shape[0],grid_hor.shape[1]), np.nan, dtype = np.float32)
        #    for i in range(grid_hor.shape[0]):
        #        for j in range(grid_hor.shape[1]):
        #            k = i * grid_hor.shape[1] + j
        #            if np.isnan(fr_intv[k]).all():
        #                h_new_all[i,j] = np.nan
        #        
        #            else:
        #                h_new_all[i,j] = fr_intv[k]
    
    return z,h_new_all

def compute_slice(cube_in, hor1,hor2, type_interpolation, shift, distance_between, num_worker):
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
        hdata3 = ((hdata2-hdata1) * const_step) + hdata1
    else:
        hdata3 = hdata1 + shift
    
    z_step_ms = cube_in.time_step / 1000
    
    cube_time = np.arange(cube_in.data_start, cube_in.data_start  + z_step_ms * cube_in.n_samples, z_step_ms)
    grid_real, grid_not = generate_fragments(cube_in.min_i, cube_in.n_i, incr_i, cube_in.min_x, cube_in.n_x, incr_x, hdata3)
    new_zr_all = np.full((hdata1.shape[0],hdata1.shape[1]), np.nan,dtype = np.float32)

    global total_frag
    total_frag = len(grid_real)
    
    with ProcessPoolExecutor(max_workers=num_worker) as executor:
        futures=[]
        for k in range(len(grid_real)):
            grid_hor3 = hdata3[grid_not[k][0]:grid_not[k][0] + grid_not[k][1],grid_not[k][2]:grid_not[k][2] + grid_not[k][3]]
            f = executor.submit(compute_fragment,k,cube_in,grid_hor3,cube_time,grid_real,type_interpolation)
            #f = executor.submit(compute_fragment,k,grid_hor3)
            f.add_done_callback(move_progress)
            futures.append(f)
            #LOG.debug(f"Submitted: {k=}")

        for f in as_completed(futures):

            try:
                z,h_new_all = f.result()
                
                #LOG.debug(f"Returned {z=}")
                new_zr_all[grid_not[z][0]:grid_not[z][0] + grid_not[z][1], grid_not[z][2]:grid_not[z][2] + grid_not[z][3]] = h_new_all
            except Exception as e:
                LOG.error(f"Exception: {e}")       
        
    new_zr_all = new_zr_all.astype('float32')
    np.nan_to_num(new_zr_all, nan=MAXFLOAT, copy=False)
    if hdata3.dtype==np.float64:
        hdata3 = hdata3.astype('float32')
    np.nan_to_num(hdata3, nan=MAXFLOAT, copy=False)
    return np.vstack([hdata3[None, :, :]] + [new_zr_all[None, :, :]])
    
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
    cube_in = job.open_input_dataset()
    hor_name1 = job.description["Horizon"][0]
    hor_name2 = job.description["Horizon"][1] if len(job.description["Horizon"])>1 else None
    type_interpolation = job.description["interpolation"]
    hor1 = job.session.get_horizon_3d(cube_in.geometry_name, hor_name1)
    hor2 = job.session.get_horizon_3d(cube_in.geometry_name, hor_name2) if hor_name2 is not None else None
    dt = compute_slice(cube_in, hor1,hor2, type_interpolation, shift, distance_between, num_worker)
    #dt = cProfile.run('compute_slice(cube_in, hor1, hor2, type_interpolation, shift, distance_between, num_worker)')
    f_out = job.session.create_attribute_2d_writer_for_cube(cube_in, job.description["New Name"], attr_name)
    f_out.write_horizon_data(dt[0]) # Not needed if the horizon data have been copied by create_attr (copy_horizon_data=True)
    f_out.write_data(dt[1])

    LOG.info(f"Processing time (s): {time.time() - tm_start}")