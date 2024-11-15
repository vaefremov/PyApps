from typing import Optional, Tuple
import logging
import numpy as np
from scipy.interpolate import interp1d
from scipy.fft import rfft, rfftfreq
from numba import njit

from di_lib import di_app
from di_lib.di_app import Context
from di_lib.seismic_cube import DISeismicCube
from di_lib.attribute import DIHorizon3D, DIAttribute2D

from typing import List

import time

logging.basicConfig(level=logging.DEBUG)
LOG = logging.getLogger(__name__)

incr_i = 100
incr_x = 100

border_correction = 3  
zero_samples = 50  

def taper_fragment(fr):
    global border_correction
    global zero_samples
    bl = np.blackman(border_correction*2+1)
    mask = np.ones(fr.shape[0], dtype=np.float32)
    mask[-border_correction:] = bl[border_correction+1:]
    res = np.hstack((np.multiply(fr, mask), np.zeros(zero_samples)))
    return res

def generate_fragments(min_i, n_i, incr_i, min_x, n_x, incr_x, hdata):
    inc_i = incr_i
    inc_x = incr_x
    res1 = [(i, j, min(n_i-1, i+inc_i-1), min(n_x-1, j+inc_x-1)) for i in range(min_i, n_i-1, inc_i) for j in range(min_x, n_x-1, inc_x)]
    res2 = [(i, j, min(hdata.shape[0]-1, i+inc_i-1), min(hdata.shape[1]-1, j+inc_x-1)) for i in range(0, hdata.shape[0]-1, inc_i) for j in range(0, hdata.shape[1]-1, inc_x)]
    return [(i[0], i[2]-i[0]+1, i[1], i[3]-i[1]+1) for i in res1], [(i[0], i[2]-i[0]+1, i[1], i[3]-i[1]+1) for i in res2]
#@njit 
def linear_interpolate(y, z, zs):
    y_out = np.zeros((y.shape[0],y.shape[1]))
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            good_idx = np.where( np.isfinite(y[i,j,:]) )
            try :
                #y_out[i,j] = interp1d(z[good_idx], y[i,j,good_idx], axis=-1, bounds_error=False )(zs[i,j])
                y_out[i,j] = np.interp(zs[i,j], z[good_idx], y[i,j,good_idx][0,:], left = np.nan, right = np.nan )
            except:
                y_out[i,j] = np.nan
    return y_out
@njit 
def mean_power(a, ind, up_sample, down_sample):
    m_p = np.zeros((a.shape[0],a.shape[1]))
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            
            if np.isnan(ind[i,j]) or np.isnan(a[i,j,int(ind[i,j])]):
                m_p[i,j] = np.nan
           
            else:
                ind_ = int(ind[i,j])
                len_a = len(a[i,j,ind_ - down_sample:ind_ + up_sample + 1])
                len_a = len_a if len_a!=0 else 1
                m_p[i,j] = np.sum(a[i,j,ind_ - down_sample:ind_ + up_sample + 1]**2) / len_a
    return m_p

@njit       
def effective_amplitude(a, ind, up_sample, down_sample):
    e_a = np.zeros((a.shape[0],a.shape[1]))
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):

            if np.isnan(ind[i,j]) or np.isnan(a[i,j,int(ind[i,j])]):
                e_a[i,j] = np.nan
           
            else:
                ind_ = int(ind[i,j])
                len_a = len(a[i,j,ind_ - down_sample:ind_ + up_sample + 1])
                len_a = len_a if len_a!=0 else 1
                e_a[i,j] = np.sqrt(np.sum(a[i,j,ind_ - down_sample:ind_ + up_sample + 1]**2)) / len_a
    return e_a
def autocorrelation_period(a,ind, up_sample, down_sample, dt):
    a_p = np.zeros((a.shape[0],a.shape[1]))
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):

            if np.isnan(ind[i,j]) or np.isnan(a[i,j,int(ind[i,j])]):
                a_p[i,j] = np.nan
           
            else:
                ind_ = int(ind[i,j])
                interval = a[i,j,ind_ - down_sample:ind_ + up_sample + 1]
                
                interval = interval[~np.isnan(interval)]
                interval = interval - np.mean(interval)
                if interval.shape[0] < 10:
                    a_p[i,j]  = np.nan
                else:
                    ind_half_period = np.argmin(np.correlate(interval, interval, 'same')[interval.shape[0]//2:]) 

                    if ind_half_period < 5 :
                        a_p[i,j]  = np.nan
                    else:
                        a_p[i,j] = ind_half_period * 2 * dt
    return a_p



def mean_freq(a, ind, up_sample, down_sample, f_min, f_max, dt):
    m_f = np.zeros((a.shape[0],a.shape[1]))
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):

            if np.isnan(ind[i,j]) or np.isnan(a[i,j,int(ind[i,j])]):
                m_f[i,j] = np.nan
           
            else:
                ind_ = int(ind[i,j])
                interval = a[i,j,ind_ - down_sample:ind_ + up_sample + 1]
                interval = interval[~np.isnan(interval)]
                len_a = interval.shape[0]
                len_a = len_a if len_a!=0 else 1
                if len_a < 5:
                    m_f[i,j]  = np.nan
                else:
                    spectr = np.abs(rfft(interval))[int(f_min * len_a * dt):int(f_max * len_a * dt)]
                    freqs = rfftfreq(len_a, dt)[int(f_min * len_a * dt):int(f_max * len_a * dt)]
                    if f_min == 0.:
                        spectr[0] = 0
                    m_f[i,j] = np.dot(spectr, freqs) / np.sum(freqs)
    return m_f

def signal_compression(a, ind, up_sample, down_sample, f_min, f_max, dt):
    global zero_samples
    s_c = np.zeros((a.shape[0],a.shape[1]))
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):

            if np.isnan(ind[i,j]) or np.isnan(a[i,j,int(ind[i,j])]):
                s_c[i,j] = np.nan
           
            else:
                ind_ = int(ind[i,j])
                interval = a[i,j,ind_ - down_sample:ind_ + up_sample + 1]
                interval = interval[~np.isnan(interval)]
                len_a = interval.shape[0]
                if len_a < 5:
                    s_c[i,j]  = np.nan
                else:
                    interval = taper_fragment(interval)
                   
                    power = (np.abs(np.fft.fft(interval))[int(f_min*(len_a + zero_samples)*dt):int(f_max*(len_a + zero_samples)*dt)])**2
                    if f_min == 0.:
                        power[0] = 0
                    s_c[i,j] = np.sum(power) / ((f_max - f_min) * np.max(power))
    return s_c

def left_spectral_area(a, ind, up_sample, down_sample, f_min, f_max, dt):
    global zero_samples
    l_sa = np.zeros((a.shape[0],a.shape[1]))
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):

            if np.isnan(ind[i,j]) or np.isnan(a[i,j,int(ind[i,j])]):
                l_sa[i,j] = np.nan
           
            else:
                ind_ = int(ind[i,j])
                interval = a[i,j,ind_ - down_sample:ind_ + up_sample + 1]
                interval = interval[~np.isnan(interval)]
                len_a = interval.shape[0]
                if len_a < 5:
                    l_sa[i,j]  = np.nan
                else:
                    interval = taper_fragment(interval)
                   
                    power = (np.abs(np.fft.fft(interval))[int(f_min*(len_a + zero_samples)*dt):int(f_max*(len_a + zero_samples)*dt)])**2
                    if f_min == 0.:
                        power[0] = 0
                    l_sa[i,j] = np.sum(power)
    return l_sa

def right_spectral_area(a, ind, up_sample, down_sample, f_min, f_max, dt):
    global zero_samples
    r_sa = np.zeros((a.shape[0],a.shape[1]))
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):

            if np.isnan(ind[i,j]) or np.isnan(a[i,j,int(ind[i,j])]):
                r_sa[i,j] = np.nan
           
            else:
                ind_ = int(ind[i,j])
                interval = a[i,j,ind_ - down_sample:ind_ + up_sample + 1]
                interval = interval[~np.isnan(interval)]
                len_a = interval.shape[0]
                if len_a < 5:
                    r_sa[i,j]  = np.nan
                else:
                    interval = taper_fragment(interval)
                   
                    power = (np.abs(np.fft.fft(interval))[int(f_min*(len_a + zero_samples)*dt):int(f_max*(len_a + zero_samples)*dt)])**2
                    if f_min == 0.:
                        power[0] = 0
                    r_sa[i,j] = np.sum(power)
    return r_sa

def compute_attribute(cube_in: DISeismicCube, hor_in: DIHorizon3D, attributes: List[str], distance_up, distance_down, min_freq, max_freq, bearing_freq) -> Optional[np.ndarray]:
    
    MAXFLOAT = float(np.finfo(np.float32).max) 
    hdata = hor_in.get_data()
    hdata = np.where((hdata>= 0.1*MAXFLOAT) | (hdata== np.inf), np.nan, hdata)
    z_step = 0.002

    cube_time = np.arange(cube_in.data_start, cube_in.data_start  + (cube_in.time_step/1000) * cube_in.n_samples, cube_in.time_step/1000)
    grid_real, grid_not = generate_fragments(cube_in.min_i, cube_in.n_i, incr_i, cube_in.min_x, cube_in.n_x, incr_x,hdata)
    # Note: Here we use the fact that since Py 3.6 dict is ordered dict
    new_zr_all = {a: np.full((hdata.shape[0],hdata.shape[1]), np.nan) for a in attributes}
    total_frag = len(grid_real)
    completed_frag = 0

    for k in range(len(grid_real)):
        grid_hor = hdata[grid_not[k][0]:grid_not[k][0] + grid_not[k][1],grid_not[k][2]:grid_not[k][2] + grid_not[k][3]]
        if np.all(np.isnan(grid_hor)) == True:
            continue
        else:
            good_idx = np.where(np.isfinite(grid_hor))
            hdata_good = grid_hor[good_idx]
            index_max = np.where((cube_time >= np.max(np.round(hdata_good)) - 1) & (cube_time <= np.max(np.round(hdata_good)) + 1))[0]
            index_min = np.where((cube_time >= np.min(np.round(hdata_good)) - 1) & (cube_time <= np.min(np.round(hdata_good)) + 1))[0]

            new_dist_up = int((distance_up)/(cube_in.time_step / 1000))
            new_dist_down = int((distance_down)/(cube_in.time_step / 1000))
            radius_samples = max(new_dist_up, new_dist_down ) 
        
            cube_time_new = cube_time[index_min[0] - new_dist_down : index_max[0] + new_dist_up]  
            
            indxs = np.round(grid_hor-cube_time_new[0])/(cube_time_new[1] - cube_time_new[0]) 

            fr = cube_in.get_fragment_z(grid_real[k][0],grid_real[k][1], grid_real[k][2],grid_real[k][3],index_min[0]-radius_samples,(index_max[0]+radius_samples) - (index_min[0]-radius_samples))

            h_new_all = {a: np.full((grid_hor.shape[0],grid_hor.shape[1]), np.nan) for a in attributes}
            if np.size(fr) == 1:
                continue
            else:
                #if grid_hor[i,j] <= 0.1 * MAXFLOAT:
                if "Amplitude" or "Pow_a_div_effective_amp" or "Abs_a_div_effective_amp" in attributes:
                    h_new_all["Amplitude"] = linear_interpolate (fr, cube_time_new, grid_hor)
         
                if "Energy" in attributes:
                    h_new_all["Energy"] = mean_power(fr, indxs, new_dist_down, new_dist_up)
                    
                if "Effective_amp" or "Pow_a_div_effective_amp" or "Abs_a_div_effective_amp" in attributes:
                    h_new_all["Effective_amp"] = effective_amplitude(fr, indxs, new_dist_down, new_dist_up)

                if "Pow_a_div_effective_amp" in attributes:
                    h_new_all["Pow_a_div_effective_amp"] = np.power(h_new_all["Amplitude"],2) / h_new_all["Effective_amp"]

                if "Abs_a_div_effective_amp" in attributes:
                    h_new_all["Abs_a_div_effective_amp"] = np.fabs(h_new_all["Amplitude"]) / h_new_all["Effective_amp"]

                if "autocorrelation_period" in attributes:
                    h_new_all["autocorrelation_period"] = autocorrelation_period(fr, indxs, new_dist_down, new_dist_up, z_step)

                if "mean_freq" in attributes:
                    h_new_all["mean_freq"] = mean_freq(fr, indxs, new_dist_down, new_dist_up, min_freq, max_freq, z_step)

                if "signal_compression" in attributes:
                    h_new_all["signal_compression"] = signal_compression(fr, indxs, new_dist_down, new_dist_up, min_freq, max_freq, z_step)

                if "left_spectral_area" or "spectral_energy" or "absorption_Ssw_Sw" or "absorption_Ssw_Sww" in attributes:
                    h_new_all["left_spectral_area"] = left_spectral_area(fr, indxs, new_dist_down, new_dist_up, min_freq,  bearing_freq, z_step)

                if "right_spectral_area" or  "spectral_energy" or "absorption_Ssw_Sw" or "absorption_Ssw_Sww" in attributes:
                    h_new_all["right_spectral_area"] = right_spectral_area(fr, indxs, new_dist_down, new_dist_up,  bearing_freq, max_freq, z_step)

                if "spectral_energy" or "absorption_Ssw_Sw" or "absorption_Ssw_Sww" in attributes:
                    h_new_all["spectral_energy"] = h_new_all["left_spectral_area"] + h_new_all["right_spectral_area"]
                    
                if "absorption_Ssw_Sw" in attributes:
                    h_new_all["absorption_Ssw_Sw"] = h_new_all["left_spectral_area"] / h_new_all["spectral_energy"] * 100

                if "absorption_Ssw_Sww" in attributes:
                    h_new_all["absorption_Ssw_Sww"] = h_new_all["left_spectral_area"] / h_new_all["right_spectral_area"]

            for a in attributes:
                new_zr_all[a][grid_not[k][0]:grid_not[k][0] + grid_not[k][1],grid_not[k][2]:grid_not[k][2] + grid_not[k][3]] = h_new_all[a]
                new_zr_all[a] = new_zr_all[a].astype('float32')
                np.nan_to_num(new_zr_all[a], nan=MAXFLOAT, copy=False)
            completed_frag += 1
            LOG.info(f"Completion: {completed_frag*100 // total_frag}")
            job.log_progress("calculation", completed_frag*100 // total_frag)
            
    np.nan_to_num(hdata, nan=MAXFLOAT, copy=False)
    return np.vstack([hdata[None, :, :]] + [new_zr[None, :, :] for new_zr in new_zr_all.values()])

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
    attributes = job.description["attributes"]
    distance_up = job.description["distance_up"]
    distance_down = job.description["distance_down"]
    min_freq = job.description["min_freq"]
    max_freq = job.description["max_freq"]
    bearing_freq = job.description["bearing_freq"]
    cube_in = job.open_input_dataset()
    hor_name = job.description["Horizon"]
    hor = job.session.get_horizon_3d(cube_in.geometry_name, hor_name)
    f_out = job.session.create_attribute_2d_writer_as_other(hor, job.description["New Name"])
    dt = compute_attribute(cube_in, hor, attributes, distance_up, distance_down, min_freq, max_freq, bearing_freq )
    f_out.write_data(dt)
    f_out.layers_names = ["T0"] + attributes

    LOG.info(f"Processing time (s): {time.time() - tm_start}")