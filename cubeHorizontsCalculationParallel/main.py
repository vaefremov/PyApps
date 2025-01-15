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

import time
import os

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)

incr_i = 100
incr_x = 100

border_correction = 3  
zero_samples = 1000  

completed_frag = 0
total_frag = 0

def move_progress(f: Future):
    global completed_frag
    if f.exception() is None:
        completed_frag += 1
        LOG.info(f"Completion: {completed_frag*100 // total_frag}")
        job.log_progress("calculation", completed_frag*100 // total_frag)   

def taper_fragment(fr):
    global border_correction
    global zero_samples
    bl = np.blackman(border_correction*2+1)
    mask = np.ones(fr.shape[0], dtype=np.float32)
    mask[-border_correction:] = bl[border_correction+1:]
    res = np.hstack((np.multiply(fr, mask), np.zeros(zero_samples)))
    return res

def generate_fragments(min_i, n_i, incr_i, min_x, n_x, incr_x,hdata):
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

def cut_intervals(y, ind1, ind2, up_sample, down_sample):
    y_out = []
    if ind2 is not None:    
        for i in range(y.shape[0]):
            for j in range(y.shape[1]):
                if np.isnan(ind1[i,j]) or np.isnan(y[i,j,:]).all() or np.isnan(ind2[i,j]):
                    y_out.append([np.nan])
                    continue
                else:
                    ind_1 = int(ind1[i,j])
                    ind_2 = int(ind2[i,j])
                    if np.isnan(y[i,j,ind_1]) or np.isnan(y[i,j,ind_2]):
                        y_out.append([np.nan])
                        continue
                    else:
                        y_out.append(y[i,j,ind_1 - up_sample:ind_2 + down_sample + 1])
    else:
        for i in range(y.shape[0]):
            for j in range(y.shape[1]):
                if np.isnan(ind1[i,j]) or np.isnan(y[i,j,:]).all():
                    y_out.append([np.nan])
                    continue
                else:
                    ind_1 = int(ind1[i,j])
                    if np.isnan(y[i,j,ind_1]):
                        y_out.append([np.nan])
                        continue
                    else:
                        y_out.append(y[i,j,ind_1 - up_sample:ind_1 + down_sample + 1])
    return y_out, None

def linear_interpolate_traces(y, c_time, ind1, ind2, gr_hor1, gr_hor2, up_sample, down_sample, t_step):
    y_out  = []
    dt_new = []
    if ind2 is not None:    
        for i in range(y.shape[0]):
            for j in range(y.shape[1]):
                if np.isnan(ind1[i,j]) or np.isnan(y[i,j,:]).all() or np.isnan(ind2[i,j]):
                    y_out.append([np.nan])
                    dt_new.append(np.nan)
                    continue
                else:
                    ind_1 = int(ind1[i,j])
                    ind_2 = int(ind2[i,j])
                    if np.isnan(y[i,j,ind_1]) or np.isnan(y[i,j,ind_2]):
                        y_out.append([np.nan])
                        dt_new.append(np.nan)
                        continue
                    else:

                        good_idx = np.where( np.isfinite(y[i,j,:]) )
                        #x0 = np.arange(gr_hor1[i,j] - up_sample*t_step , gr_hor2[i,j] + (down_sample+1)*t_step,t_step)
                        x0 = np.linspace(gr_hor1[i,j] - up_sample*t_step , gr_hor2[i,j] + (down_sample+1)*t_step,50)
                        try:
                            y_out.append(np.interp(x0, c_time[good_idx], y[i,j,good_idx][0,:], left = np.nan, right = np.nan ))
                            dt_new.append((x0[1]-x0[0])*0.001)
                        except:
                            y_out.append([np.nan])
                            dt_new.append(np.nan)
    else:                    
        for i in range(y.shape[0]):
            for j in range(y.shape[1]):
                if np.isnan(gr_hor1[i,j]) or np.isnan(y[i,j,:]).all():
                    y_out.append([np.nan])
                    dt_new.append(np.nan)
                    continue
                else:
                    ind_1 = int(ind1[i,j])
                    if np.isnan(y[i,j,ind_1]):
                        y_out.append([np.nan])
                        dt_new.append(np.nan)
                        continue
                    else:
                        good_idx = np.where( np.isfinite(y[i,j,:]) )
                        #x0 = np.arange(gr_hor1[i,j] - up_sample*t_step , gr_hor1[i,j] + (down_sample+1)*t_step,t_step)
                        x0 = np.linspace(gr_hor1[i,j] - up_sample*t_step , gr_hor1[i,j] + (down_sample+1)*t_step,50)
                        try :
                            y_out.append(np.interp(x0, c_time[good_idx], y[i,j,good_idx][0,:], left = np.nan, right = np.nan ))
                            dt_new.append((x0[1]-x0[0])*0.001)
                        except:
                            y_out.append([np.nan])
                            dt_new.append(np.nan)
                            continue
    return y_out, dt_new
def cubic_interpolate_traces(y, c_time, ind1, ind2, gr_hor1, gr_hor2, up_sample, down_sample, t_step):
    y_out = []
    dt_new = []
    if ind2 is not None:    
        for i in range(y.shape[0]):
            for j in range(y.shape[1]):
                if np.isnan(ind1[i,j]) or np.isnan(y[i,j,:]).all() or np.isnan(ind2[i,j]):
                    y_out.append([np.nan])
                    dt_new.append(np.nan)
                    continue
                else:
                    ind_1 = int(ind1[i,j])
                    ind_2 = int(ind2[i,j])
                    if np.isnan(y[i,j,ind_1]) or np.isnan(y[i,j,ind_2]):
                        y_out.append([np.nan])
                        dt_new.append(np.nan)
                        continue
                    else:

                        good_idx = np.where( np.isfinite(y[i,j,:]) )
                        #x0 = np.arange(gr_hor1[i,j] - up_sample*t_step , gr_hor2[i,j] + (down_sample+1)*t_step,t_step)
                        x0 = np.linspace(gr_hor1[i,j] - up_sample*t_step , gr_hor2[i,j] + (down_sample+1)*t_step,50)
                        try:
                            y_out.append(CubicSpline(c_time[good_idx], y[i,j,good_idx][0,:], extrapolate=False )(x0))
                            dt_new.append((x0[1]-x0[0])*0.001)
                        except:
                            y_out.append([np.nan])
                            dt_new.append(np.nan)
    else:                    
        for i in range(y.shape[0]):
            for j in range(y.shape[1]):
                if np.isnan(gr_hor1[i,j]) or np.isnan(y[i,j,:]).all():
                    y_out.append([np.nan])
                    dt_new.append(np.nan)
                    continue
                else:
                    ind_1 = int(ind1[i,j])
                    if np.isnan(y[i,j,ind_1]):
                        y_out.append([np.nan])
                        dt_new.append(np.nan)
                        continue
                    else:
                        good_idx = np.where( np.isfinite(y[i,j,:]) )
                        #x0 = np.arange(gr_hor1[i,j] - up_sample*t_step , gr_hor1[i,j] + (down_sample+1)*t_step,t_step)
                        x0 = np.linspace(gr_hor1[i,j] - up_sample*t_step , gr_hor1[i,j] + (down_sample+1)*t_step,50)
                        try :
                            y_out.append(CubicSpline(c_time[good_idx], y[i,j,good_idx][0,:], extrapolate=False )(x0))
                            dt_new.append((x0[1]-x0[0])*0.001)
                        except:
                            y_out.append([np.nan])
                            dt_new.append(np.nan)
                            continue
    return y_out, dt_new


def mean_amplitude(a,grid_size):
    m_a = np.full((grid_size[0],grid_size[1]), np.nan)
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            k = i*grid_size[1] + j
            if np.isnan(a[k]).all():
                m_a[i,j] = np.nan
            else:
                m_a[i,j]= np.nanmean(a[k])
    return m_a

def sum_amplitude(a,grid_size):
    s_a = np.full((grid_size[0],grid_size[1]), np.nan)
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            k=i*grid_size[1] + j
            if np.isnan(a[k]).all():
                s_a[i,j] = np.nan
            else:
                s_a[i,j]= np.nansum(a[k])
    return s_a

#@njit 
def mean_power(a,grid_size):
    m_p = np.full((grid_size[0],grid_size[1]), np.nan)
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            k= i * grid_size[1] + j
            if np.isnan(a[k]).all():
                m_p[i,j] = np.nan
           
            else:
                a_i = a[k]
                a_i = a_i[~np.isnan(a_i)]
                len_a = len(a_i)
                len_a = len_a if len_a!=0 else 1
                m_p[i,j] = np.sum((a_i**2)) / len_a
    return m_p
       
def effective_amplitude(a,grid_size):
    e_a = np.full((grid_size[0],grid_size[1]), np.nan)
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            k = i * grid_size[1] + j
            if np.isnan(a[k]).all():
                e_a[i,j] = np.nan
           
            else:
                a_i = a[k]
                a_i = a_i[~np.isnan(a_i)]
                len_a = len(a_i)
                len_a = len_a if len_a!=0 else 1
                e_a[i,j] = np.sqrt(np.sum(a_i**2)) / len_a
    return e_a

def autocorrelation_period(a, grid_size, dt, dt_new):
    if dt_new is None:
        dt_new = np.ones(len(a))*dt
    a_p = np.full((grid_size[0],grid_size[1]), np.nan)
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            k = i * grid_size[1] + j
            if np.isnan(a[k]).all():
                a_p[i,j] = np.nan
            else:
                interval = a[k]
                interval = interval[~np.isnan(interval)]
                interval = interval - np.mean(interval)
                if interval.shape[0] < 5:
                    a_p[i,j]  = np.nan
                else:
                    corr = np.correlate(interval, interval, 'full')[interval.shape[0]-1:]
                    ind_half_period = np.argmax(np.sign(np.diff(corr))) 

                    if ind_half_period < 5 or ind_half_period >= interval.shape[0] / 2:
                        a_p[i,j]  = np.nan
                    else:
                        a_p[i,j] = ind_half_period * 2 * dt_new[k]
    return a_p

def mean_freq(s_all, freqs, grid_size, f_min, f_max, dt, dt_new):
    global zero_samples
    if dt_new is None:
        dt_new = np.ones(len(s_all))*dt
    m_f = np.full((grid_size[0],grid_size[1]), np.nan)
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            k = i * grid_size[1] + j
            if np.isnan(s_all[k]).all():
                m_f[i,j] = np.nan
            else:
                interval = s_all[k]
                if len(interval)==1:
                    m_f[i,j] = np.nan
                else:
                    len_a = interval.shape[0]
                    len_a = len_a if len_a!=0 else 1
                    if len_a < 2:
                        m_f[i,j]  = np.nan
                    else:
                        s = interval[int(f_min*2*len_a*dt_new[k]):int(f_max*2*len_a*dt_new[k])]
                        freq = freqs[k][int(f_min*2*len_a*dt_new[k]):int(f_max*2*len_a*dt_new[k])]
                        if f_min == 0.:
                            s[0] = 0
                        m_f[i,j] = np.dot(s, freq) / np.sum(s)
    return m_f

def signal_compression(s_all,grid_size, f_min, f_max, dt, dt_new):
    global zero_samples
    if dt_new is None:
        dt_new = np.ones(len(s_all))*dt
    s_c = np.full((grid_size[0],grid_size[1]), np.nan)
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            k = i *  grid_size[1] + j
            if np.isnan(s_all[k]).all():
                s_c[i,j] = np.nan
            else:
                interval = s_all[k]
                len_a = interval.shape[0]
                if len_a < 2:
                    s_c[i,j]  = np.nan
                else:
                    power = (interval[int(f_min*2*len_a*dt_new[k]):int(f_max*2*len_a*dt_new[k])])**2
                    if f_min == 0.:
                        power[0] = 0
                    s_c[i,j] = np.sum(power) / ((f_max - f_min) * np.max(power))
    return s_c

def left_spectral_area(s_all,grid_size, f_min, f_max, dt, dt_new):
    global zero_samples
    if dt_new is None:
        dt_new = np.ones(len(s_all))*dt
    l_sa = np.full((grid_size[0],grid_size[1]), np.nan)
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            k = i * grid_size[1] + j
            if np.isnan(s_all[k]).all():
                l_sa[i,j] = np.nan
            else:
                interval = s_all[k]
                len_a = interval.shape[0]
                if len_a < 2:
                    l_sa[i,j]  = np.nan
                else:
                    power = (interval[int(f_min*2*len_a*dt_new[k]):int(f_max*2*len_a*dt_new[k])])**2
                    if f_min == 0.:
                        power[0] = 0
                    l_sa[i,j] = np.sum(power)
    return l_sa


def right_spectral_area(s_all,grid_size, f_min, f_max, dt, dt_new):
    global zero_samples
    if dt_new is None:
        dt_new = np.ones(len(s_all))*dt
    r_sa = np.full((grid_size[0],grid_size[1]), np.nan)
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            k = i * grid_size[1] + j
            if np.isnan(s_all[k]).all():
                r_sa[i,j] = np.nan
            else:
                interval = s_all[k]
                len_a = interval.shape[0]
                if len_a < 2:
                    r_sa[i,j]  = np.nan
                else:
                    power = (interval[int(f_min*2*len_a*dt_new[k]):int(f_max*2*len_a*dt_new[k])])**2
                    if f_min == 0.:
                        power[0] = 0
                   
                    r_sa[i,j] = np.sum(power)
    return r_sa
def Fourier_transform(a, dt, dt_new):
    global zero_samples
    global border_correction
    if dt_new is None:
        dt_new = np.ones(len(a))*dt
    spectr = []
    freqs  = []
    for k in range(len(a)):
        interval = a[k]
        if len(interval)==1:
            spectr.append([np.nan])
            freqs.append([np.nan])
        else:
            interval = interval[~np.isnan(interval)]
            len_a = interval.shape[0]
            
            if len_a < 2:
                spectr.append([np.nan])
                freqs.append([np.nan])
            else:
                interval = taper_fragment(interval)
                spectr.append(np.abs(rfft(interval)))
                freqs.append(rfftfreq(len_a + zero_samples, dt_new[k]))
    return spectr, freqs

def compute_fragment(z, cube_in, distance_up, distance_down, min_freq, max_freq, bearing_freq, grid_hor1, grid_hor2, cube_time, grid_real, z_step_ms, hor_in2, type_interpolation, attributes):
    MAXFLOAT = float(np.finfo(np.float32).max) 
    #LOG.info(f"Starting job {os.getpid()} {z=} {(grid_hor1.shape, grid_hor2.shape, cube_time)}")
    tm_start = time.time()
    h_new_all = {a: np.full((grid_hor1.shape[0],grid_hor1.shape[1]), np.nan, dtype = np.float32) for a in attributes}
    if np.all(np.isnan(grid_hor1)) == True :
        #LOG.info(f"Finish job (all nones hor1) {os.getpid()}  {z=} {(grid_hor1.shape, grid_hor2.shape, cube_time)}")
        return z, h_new_all
    
    elif np.all(np.isnan(grid_hor2)) == True if grid_hor2 is not None else False:
        #LOG.info(f"Finish job (all nones hor2) {os.getpid()}  {z=} {(grid_hor1.shape, grid_hor2.shape, cube_time)}")
        return z, h_new_all

    else:
        
        z_step = z_step_ms/1000
        up_sample   = int(distance_up / z_step_ms )
        down_sample = int(distance_down / z_step_ms )
        if hor_in2 is not None:
            index_max = np.argmin(np.abs(cube_time-np.nanmax(np.round(grid_hor2))))
            index_min = np.argmin(np.abs(cube_time-np.nanmin(np.round(grid_hor1))))
        else:
            index_max = np.argmin(np.abs(cube_time-np.nanmax(np.round(grid_hor1))))
            index_min = np.argmin(np.abs(cube_time-np.nanmin(np.round(grid_hor1))))
        cube_time_new = cube_time[index_min - up_sample - 3:index_max + down_sample + 3]

        fr = cube_in.get_fragment_z(grid_real[z][0],grid_real[z][1], grid_real[z][2],grid_real[z][3],index_min-up_sample-3,(index_max+down_sample+3) - (index_min-up_sample-3))
        fr = np.where((fr>= 0.1*MAXFLOAT) | (fr== np.inf), np.nan, fr)
        indxs1 = np.round((grid_hor1-cube_time_new[0])/(cube_time_new[1] - cube_time_new[0]))
        indxs2 = np.round((grid_hor2-cube_time_new[0])/(cube_time_new[1] - cube_time_new[0])) if hor_in2 is not None else None
        h_new_all = {a: np.full((grid_hor1.shape[0],grid_hor1.shape[1]), np.nan, dtype = np.float32) for a in attributes}
    
        if type_interpolation == "no interpolation":
            fr_intv, dt_sec_new = cut_intervals(fr, indxs1, indxs2, up_sample, down_sample)
        if type_interpolation == "linear":
            fr_intv, dt_sec_new = linear_interpolate_traces(fr, cube_time_new, indxs1, indxs2, grid_hor1, grid_hor2, up_sample, down_sample, z_step_ms)
        if type_interpolation == "cubic spline":
            fr_intv, dt_sec_new =  cubic_interpolate_traces(fr, cube_time_new, indxs1, indxs2, grid_hor1, grid_hor2, up_sample, down_sample, z_step_ms)
        fr_size = fr.shape
        if np.size(fr) == 1:
            #LOG.info(f"Finish job (fr = 1) {os.getpid()}  {z=} {(grid_hor1.shape, grid_hor2.shape, cube_time)}")
            return z, h_new_all
        else:
            
            if "signal_compression" or "mean_freq" or "right_spectral_area" or "left_spectral_area" or "spectral_energy" or "absorption_Ssw_Sw" or "absorption_Ssw_Sww" in attributes:
                spectr, freqs = Fourier_transform(fr_intv, z_step, dt_sec_new)
           
            if "Amplitude" or "Pow_a_div_effective_amp" or "Abs_a_div_effective_amp" in attributes:
                h_new_all["Amplitude"] = linear_interpolate (fr, cube_time_new, grid_hor1)
        
            if "Energy" in attributes:
                h_new_all["Energy"] = mean_power(fr_intv, fr_size)
                
            if "Effective_amp" or "Pow_a_div_effective_amp" or "Abs_a_div_effective_amp" in attributes:
                h_new_all["Effective_amp"] = effective_amplitude(fr_intv, fr_size)

            if "Pow_a_div_effective_amp" in attributes:
                h_new_all["Pow_a_div_effective_amp"] = np.power(h_new_all["Amplitude"],2) / h_new_all["Effective_amp"]

            if "mean_amplitude" in attributes:
                h_new_all["mean_amplitude"] = mean_amplitude(fr_intv, fr_size)

            if "sum_amplitude" in attributes:
                h_new_all["sum_amplitude"] = sum_amplitude(fr_intv, fr_size)

            if "Abs_a_div_effective_amp" in attributes:
                h_new_all["Abs_a_div_effective_amp"] = np.fabs(h_new_all["Amplitude"]) / h_new_all["Effective_amp"]

            if "autocorrelation_period" in attributes:
                h_new_all["autocorrelation_period"] = autocorrelation_period(fr_intv, fr_size, z_step, dt_sec_new)

            if "mean_freq" in attributes:
                h_new_all["mean_freq"] = mean_freq(spectr, freqs, fr_size, min_freq, max_freq, z_step, dt_sec_new)
               
            if "signal_compression" in attributes:
                h_new_all["signal_compression"] = signal_compression(spectr, fr_size, min_freq, max_freq, z_step, dt_sec_new)

            if "left_spectral_area" or "spectral_energy" or "absorption_Ssw_Sw" or "absorption_Ssw_Sww" in attributes:
                h_new_all["left_spectral_area"] = left_spectral_area(spectr, fr_size, min_freq,  bearing_freq, z_step, dt_sec_new)

            if "right_spectral_area" or  "spectral_energy" or "absorption_Ssw_Sw" or "absorption_Ssw_Sww" in attributes:
                h_new_all["right_spectral_area"] = right_spectral_area(spectr, fr_size, bearing_freq, max_freq, z_step, dt_sec_new)

            if "spectral_energy" or "absorption_Ssw_Sw" or "absorption_Ssw_Sww" in attributes:
                h_new_all["spectral_energy"] = h_new_all["left_spectral_area"] + h_new_all["right_spectral_area"]
                
            if "absorption_Ssw_Sw" in attributes:
                h_new_all["absorption_Ssw_Sw"] = h_new_all["left_spectral_area"] / h_new_all["spectral_energy"] * 100

            if "absorption_Ssw_Sww" in attributes:
                h_new_all["absorption_Ssw_Sww"] = h_new_all["left_spectral_area"] / h_new_all["right_spectral_area"]
            #LOG.info(f"Finish job {os.getpid()} for {z=} in {time.time()-tm_start}s {(grid_hor1.shape, grid_hor2.shape, cube_time)}")
            return z,h_new_all

def compute_attribute(cube_in: DISeismicCube, hor_in1: DIHorizon3D, hor_in2: DIHorizon3D, attributes: List[str], type_interpolation, distance_up, distance_down, min_freq, max_freq, bearing_freq,num_worker) -> Optional[np.ndarray]:
    MAXFLOAT = float(np.finfo(np.float32).max) 
    hdata01 = hor_in1.get_data()
    hdata01 = np.where((hdata01>= 0.1*MAXFLOAT) | (hdata01== np.inf), np.nan, hdata01)
    hdata1 = np.full((cube_in.n_i-cube_in.min_i,cube_in.n_x-cube_in.min_x), np.nan, dtype = np.float32)

    mask = np.mgrid[cube_in.min_i:cube_in.n_i,cube_in.min_x:cube_in.n_x]
    mask1 = np.mgrid[hor1.min_i:hor1.min_i+hor1.n_i,hor1.min_x:hor1.min_x+hor1.n_x]

    loar1 = np.where((mask1[0]>=cube_in.min_i) & (mask1[0]<=cube_in.n_i),True,False) & np.where((mask1[1]>=cube_in.min_x) & (mask1[1]<=cube_in.n_x),True,False)
    loar1h = np.where((mask[0]>=hor1.min_i) & (mask[0]<=hor1.min_i+hor1.n_i),True,False) & np.where((mask[1]>=hor1.min_x) & (mask[1]<=hor1.min_x+hor1.n_x),True,False)
    hdata1[loar1h] = hdata01[loar1]

    hdata02 = hor_in2.get_data() if hor_in2 is not None else None
    if hdata02 is not None:
        
        hdata2 = np.full((cube_in.n_i-cube_in.min_i,cube_in.n_x-cube_in.min_x), np.nan,dtype = np.float32)
        hdata02 = np.where((hdata02>= 0.1*MAXFLOAT) | (hdata02== np.inf), np.nan, hdata02)
        mask2 = np.mgrid[hor2.min_i:hor2.min_i+hor2.n_i,hor2.min_x:hor2.min_x+hor2.n_x]

        loar2 = np.where((mask2[0]>=cube_in.min_i) & (mask2[0]<=cube_in.n_i),True,False) & np.where((mask2[1]>=cube_in.min_x) & (mask2[1]<=cube_in.n_x),True,False)
        loar2h = np.where((mask[0]>=hor2.min_i) & (mask[0]<=hor2.min_i+hor2.n_i),True,False) & np.where((mask[1]>=hor2.min_x) & (mask[1]<=hor2.min_x+hor2.n_x),True,False)
        hdata2[loar2h] = hdata02[loar2]
        if np.nanmean(hdata2) <= np.nanmean(hdata1):
            hdata1, hdata2 = hdata2, hdata1
    z_step_ms = cube_in.time_step / 1000

    cube_time = np.arange(cube_in.data_start, cube_in.data_start  + z_step_ms * cube_in.n_samples, z_step_ms)
    grid_real, grid_not = generate_fragments(cube_in.min_i, cube_in.n_i, incr_i, cube_in.min_x, cube_in.n_x, incr_x,hdata1)
    # Note: Here we use the fact that since Py 3.6 dict is ordered dict
    new_zr_all = {a: np.full((hdata1.shape[0],hdata1.shape[1]), np.nan,dtype = np.float32) for a in attributes}

    global total_frag
    total_frag = len(grid_real)
    
    with ProcessPoolExecutor(max_workers=num_worker) as executor:
        futures=[]
        for k in range(len(grid_real)):
            grid_hor1 = hdata1[grid_not[k][0]:grid_not[k][0] + grid_not[k][1],grid_not[k][2]:grid_not[k][2] + grid_not[k][3]]
            grid_hor2 = hdata2[grid_not[k][0]:grid_not[k][0] + grid_not[k][1],grid_not[k][2]:grid_not[k][2] + grid_not[k][3]] if hor_in2 is not None else None
            f = executor.submit(compute_fragment,k,cube_in,distance_up,distance_down,min_freq, max_freq,bearing_freq,grid_hor1,grid_hor2,cube_time,grid_real,z_step_ms,hor_in2,type_interpolation,attributes)
            f.add_done_callback(move_progress)
            futures.append(f)
            LOG.info(f"Submitted: {k=}")

        # completed_frag = 0
        for f in as_completed(futures):

            try:
                z,h_new_all = f.result()
                
                LOG.info(f"Returned {z=}")
                for a in attributes:
                    new_zr_all[a][grid_not[z][0]:grid_not[z][0] + grid_not[z][1], grid_not[z][2]:grid_not[z][2] + grid_not[z][3]] = h_new_all[a]
                LOG.info(f"After writing to new_zr_all {z=}")
            except Exception as e:
                LOG.info(f"Exception: {e}")       
            # completed_frag += 1
            # LOG.info(f"Completion: {completed_frag*100 // total_frag}")
            # job.log_progress("calculation", completed_frag*100 // total_frag) 
        
    for a in attributes:
        new_zr_all[a] = new_zr_all[a].astype('float32')
        np.nan_to_num(new_zr_all[a], nan=MAXFLOAT, copy=False)
    if hdata1.dtype==np.float64:
        hdata1 = hdata1.astype('float32')
    np.nan_to_num(hdata1, nan=MAXFLOAT, copy=False)
    return np.vstack([hdata1[None, :, :]] + [new_zr[None, :, :] for new_zr in new_zr_all.values()])
    
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
    attributes = job.description["attributes"]
    distance_up = job.description["distance_up"]
    distance_down = job.description["distance_down"]
    min_freq = job.description["min_freq"]
    max_freq = job.description["max_freq"]
    bearing_freq = job.description["bearing_freq"]
    num_worker = job.description["num_worker"]
    cube_in = job.open_input_dataset()
    hor_name1 = job.description["Horizon"][0]
    hor_name2 = job.description["Horizon"][1] if len(job.description["Horizon"])>1 else None
    type_interpolation = job.description["interpolation"]
    hor1 = job.session.get_horizon_3d(cube_in.geometry_name, hor_name1)
    hor2 = job.session.get_horizon_3d(cube_in.geometry_name, hor_name2) if hor_name2 is not None else None
    dt = compute_attribute(cube_in, hor1,hor2, attributes, type_interpolation, distance_up, distance_down, min_freq, max_freq, bearing_freq, num_worker)
    for i,attr_name in enumerate(attributes, 1):
        f_out = job.session.create_attribute_2d_writer_for_cube(cube_in, job.description["New Name"], attr_name)
        f_out.write_horizon_data(dt[0]) # Not needed if the horizon data have been copied by create_attr (copy_horizon_data=True)
        f_out.write_data(dt[i])

    LOG.info(f"Processing time (s): {time.time() - tm_start}")