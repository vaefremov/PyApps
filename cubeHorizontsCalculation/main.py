from typing import Optional, Tuple
import logging
import numpy as np
from scipy.interpolate import interp1d

from di_lib import di_app
from di_lib.di_app import Context
from di_lib.seismic_cube import DISeismicCube
from di_lib.attribute import DIHorizon3D, DIAttribute2D

import time

logging.basicConfig(level=logging.DEBUG)
LOG = logging.getLogger(__name__)

incr_i = 100
incr_x = 100

def generate_fragments(min_i, n_i, incr_i, min_x, n_x, incr_x, hdata):
    inc_i = incr_i
    inc_x = incr_x
    res1 = [(i, j, min(n_i-1, i+inc_i-1), min(n_x-1, j+inc_x-1)) for i in range(min_i, n_i-1, inc_i) for j in range(min_x, n_x-1, inc_x)]
    res2 = [(i, j, min(hdata.shape[0]-1, i+inc_i-1), min(hdata.shape[1]-1, j+inc_x-1)) for i in range(0, hdata.shape[0]-1, inc_i) for j in range(0, hdata.shape[1]-1, inc_x)]
    return [(i[0], i[2]-i[0]+1, i[1], i[3]-i[1]+1) for i in res1], [(i[0], i[2]-i[0]+1, i[1], i[3]-i[1]+1) for i in res2]

def linear_interpolate(y, z, zs):
    good_idx = np.where( np.isfinite(y) )
    try :
        y_out = interp1d(z[good_idx], y[good_idx], axis=-1, bounds_error=False )(zs)
        return y_out
    except:
        return np.nan

def compute_attribute(cube_in: DISeismicCube, hor_in: DIHorizon3D, attribute, distance_up, distance_down) -> Optional[np.ndarray]:
    
    MAXFLOAT = float(np.finfo(np.float32).max) 
    hdata = hor_in.get_data()
    hdata = np.where((hdata>= 0.1*MAXFLOAT) | (hdata== np.inf), np.nan, hdata)

    cube_time = np.arange(cube_in.data_start, cube_in.data_start  + (cube_in.time_step/1000) * cube_in.n_samples, cube_in.time_step/1000)
    grid_real, grid_not = generate_fragments(cube_in.min_i, cube_in.n_i, incr_i, cube_in.min_x, cube_in.n_x, incr_x,hdata)
    new_zr = np.full((hdata.shape[0],hdata.shape[1]), np.nan)
    total_frag = len(grid_real)
    completed_frag = 0

    for k in range(len(grid_real)):
        grid_hor = hdata[grid_not[k][0]:grid_not[k][0] + grid_not[k][1],grid_not[k][2]:grid_not[k][2] + grid_not[k][3]]
        if np.all(np.isnan(grid_hor)) == True:
            continue
        else:
            good_idx = np.where(np.isfinite(grid_hor))
            hdata1 = grid_hor[good_idx]
            index_max = np.where((cube_time >= np.max(np.round(hdata1)) - 1) & (cube_time <= np.max(np.round(hdata1)) + 1))[0]
            index_min = np.where((cube_time >= np.min(np.round(hdata1)) - 1) & (cube_time <= np.min(np.round(hdata1)) + 1))[0]
            cube_time_new = cube_time[index_min[0]-5:index_max[0]+5]
            fr = cube_in.get_fragment_z(grid_real[k][0],grid_real[k][1], grid_real[k][2],grid_real[k][3],index_min[0]-5,(index_max[0]+5) - (index_min[0]-5))
            h_new = np.full((grid_hor.shape[0],grid_hor.shape[1]), np.nan)
            for i in range(grid_hor.shape[0]):
                for j in range(grid_hor.shape[1]):
                    if grid_hor[i,j] <= 0.1 * MAXFLOAT:
                        if np.size(fr) == 1:
                            continue
                        else:
                            if "Amplitude" in attribute:
                                h_new[i,j] = linear_interpolate(fr[i,j,:], cube_time_new, grid_hor[i,j])
                            if "Energy" in attribute:
                                ind = int(np.round(grid_hor[i,j]-cube_time_new[0])/(cube_time_new[1] - cube_time_new[0]))
                                new_dist_up = int((distance_up)/(cube_in.time_step / 1000))
                                new_dist_down = int((distance_down)/(cube_in.time_step / 1000))
                                h_new[i,j] = (np.sum(fr[i,j,ind - new_dist_down:ind + new_dist_up]**2))/len(fr[i,j,ind - new_dist_down:ind + new_dist_up])
    
            new_zr[grid_not[k][0]:grid_not[k][0] + grid_not[k][1],grid_not[k][2]:grid_not[k][2] + grid_not[k][3]] = h_new
            new_zr = new_zr.astype('float32')
            completed_frag += 1
            LOG.info(f"Completion: {completed_frag*100 // total_frag}")
            job.log_progress("calculation", completed_frag*100 // total_frag)
            np.nan_to_num(new_zr, nan=MAXFLOAT, copy=False)
 
    return new_zr

class cubeHorizontsCalculation(di_app.DiAppSeismic3D):
    def __init__(self) -> None:
        super().__init__(in_name_par="Input Seismic3D Names", 
                out_name_par="New Name", out_names=[])

    def compute(self, f_in_tup: Tuple[np.ndarray], context: Context) -> Tuple:
        raise NotImplementedError("Shouldn't be called in this application!")

if __name__ == "__main__":
    LOG.debug(f"Starting job ExampleHor1")
    tm_start = time.time()
    job = cubeHorizontsCalculation()
    attribute = job.description["attribute"]
    distance_up = job.description["distance_up"]
    distance_down = job.description["distance_down"]
    cube_in = job.open_input_dataset()
    hor_name = job.description["Horizon"]
    hor = job.session.get_horizon_3d(cube_in.geometry_name, hor_name)
    f_out = job.session.create_horizon_3d_writer_as_other(hor, job.description["New Name"])
    dt = compute_attribute(cube_in, hor, attribute, distance_up, distance_down)
    f_out.write_data(dt)

    LOG.info(f"Processing time (s): {time.time() - tm_start}")