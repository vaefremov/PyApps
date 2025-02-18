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

def move_progress(f: Future):
    global completed_frag
    if f.exception() is None:
        completed_frag += 1
        LOG.info(f"Completion: {completed_frag*100 // total_frag}")
        job.log_progress("calculation", completed_frag*100 // total_frag)
        
def distance_to_center(fragment,center_x, center_y):
    # Находим фрагменты ближайшие к центру
    x, dx, y, dy = fragment
    center_x_frag = x + dx // 2
    center_y_frag = y + dy // 2
    return np.sqrt((center_x - center_x_frag) ** 2 + (center_y - center_y_frag) ** 2)
        
def cut_intervals(y, ind1, ind2):
    y_out = np.full((y.shape[0],y.shape[1],y.shape[2]), np.nan,dtype = np.float32)
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            if np.isnan(ind1[i,j]) or np.isnan(y[i,j,:]).all() or np.isnan(ind2[i,j]):
                continue
            else:
                ind_1 = int(ind1[i,j])
                ind_2 = int(ind2[i,j])
                if np.isnan(y[i,j,ind_1]) or np.isnan(y[i,j,ind_2]):
                    continue
                else:
                    y_out[i,j,ind_1:ind_2] = y[i,j,ind_1:ind_2]
    return y_out.flatten()
        
def value_distribution(y_values, pocket):
    b1 = np.zeros(len(pocket), dtype=int)
    pocket_array = np.array(pocket)
    if len(y_values) == 1 and np.isnan(y_values[0]):
        return []
    else:
        y_values = y_values[~np.isnan(y_values)]

        # Ищем индексы ближайших значений
        idx_right = np.searchsorted(pocket_array, y_values, side='right')
        idx_right = np.clip(idx_right, 0, len(pocket_array) - 1)
        idx_left = np.clip(idx_right - 1, 0, len(pocket_array) - 1)

        idx_closest = np.where(pocket_array[idx_left] <= y_values, idx_left, idx_right)
        nearest_values = pocket_array[idx_closest]

        unique, counts = np.unique(nearest_values, return_counts=True)

        b1[np.searchsorted(pocket_array, unique)] += counts
    
    return b1

def auto_round(num, sig_figs=3):
    if num == 0:
        return 0
    order = math.floor(math.log10(abs(num)))
    decimal_places = max(sig_figs - order - 1, 0)
    return round(num, decimal_places)

def generate_fragments(min_i, n_i, incr_i, min_x, n_x, incr_x,hdata):
    inc_i = incr_i
    inc_x = incr_x
    res1 = [(i, j, min(n_i-1, i+inc_i-1), min(n_x-1, j+inc_x-1)) for i in range(min_i, n_i-1, inc_i) for j in range(min_x, n_x-1, inc_x)]
    res2 = [(i, j, min(hdata.shape[0]-1, i+inc_i-1), min(hdata.shape[1]-1, j+inc_x-1)) for i in range(0, hdata.shape[0]-1, inc_i) for j in range(0, hdata.shape[1]-1, inc_x)]
    return [(i[0], i[2]-i[0]+1, i[1], i[3]-i[1]+1) for i in res1], [(i[0], i[2]-i[0]+1, i[1], i[3]-i[1]+1) for i in res2]

def compute_fragment(z,cube_in,grid_hor1,grid_hor2,cube_time,grid_real,pocket,n,mean,M2):
    MAXFLOAT = float(np.finfo(np.float32).max) 
    if np.all(np.isnan(grid_hor1)) == True :
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,n,mean,M2
    else:
        index_max = np.argmin(np.abs(cube_time-np.nanmax(np.round(grid_hor2))))
        index_min = np.argmin(np.abs(cube_time-np.nanmin(np.round(grid_hor1))))
        cube_time_new = cube_time[index_min-3:index_max+3]
        indxs1 = np.round((grid_hor1-cube_time_new[0])/(cube_time_new[1] - cube_time_new[0]))
        indxs2 = np.round((grid_hor2-cube_time_new[0])/(cube_time_new[1] - cube_time_new[0]))

        fr = cube_in.get_fragment_z(grid_real[z][0],grid_real[z][1], grid_real[z][2],grid_real[z][3],index_min-3,((index_max+3)-(index_min-3)))
        if fr is None or np.all(np.isnan(fr)) == True:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,n,mean,M2
        else:
            fr = np.where((fr>= 0.1*MAXFLOAT) | (fr== np.inf), np.nan, fr)
            new_fr = cut_intervals(fr,indxs1,indxs2)
            new_fr = new_fr[~np.isnan(new_fr)]
            for value in new_fr:
                n += 1
                delta = value - mean
                mean += delta / n
                M2 += delta * (value - mean)
            fr_min = np.nanmin( new_fr)
            fr_max = np.nanmax( new_fr)
            fr_mean = np.nanmean( new_fr)
            fr_median = np.nanmedian( new_fr)
            if len(pocket) != 0:
                raspr_count = value_distribution(new_fr, pocket)
            else:
                raspr_count = []

    return fr_min, fr_max, fr_mean, fr_median, pocket,raspr_count,n,mean,M2

def compute_slice(cube_in, hor1, hor2,num_worker):
    new_fr_min = []
    new_fr_max = []
    new_fr_mean = []
    new_fr_median = []
    pocket = []
    n = 0
    mean = 0.0
    M2 = 0.0
    
    MAXFLOAT = float(np.finfo(np.float32).max) 
    hdata01 = hor1.get_data()
    hdata01 = np.where((hdata01>= 0.1*MAXFLOAT) | (hdata01== np.inf), np.nan, hdata01)
    hdata1 = np.full((cube_in.n_i-cube_in.min_i,cube_in.n_x-cube_in.min_x), np.nan, dtype = np.float32)
    mask = np.mgrid[cube_in.min_i:cube_in.n_i,cube_in.min_x:cube_in.n_x]
    mask1 = np.mgrid[hor1.min_i:hor1.min_i+hor1.n_i,hor1.min_x:hor1.min_x+hor1.n_x]
    loar1 = np.where((mask1[0]>=cube_in.min_i) & (mask1[0]<cube_in.n_i),True,False) & np.where((mask1[1]>=cube_in.min_x) & (mask1[1]<cube_in.n_x),True,False)
    loar1h = np.where((mask[0]>=hor1.min_i) & (mask[0]<hor1.min_i+hor1.n_i),True,False) & np.where((mask[1]>=hor1.min_x) & (mask[1]<hor1.min_x+hor1.n_x),True,False)
    hdata1[loar1h] = hdata01[loar1]
    
    hdata02 = hor2.get_data()
    hdata2 = np.full((cube_in.n_i-cube_in.min_i,cube_in.n_x-cube_in.min_x), np.nan,dtype = np.float32)
    hdata02 = np.where((hdata02>= 0.1*MAXFLOAT) | (hdata02== np.inf), np.nan, hdata02)
    mask2 = np.mgrid[hor2.min_i:hor2.min_i+hor2.n_i,hor2.min_x:hor2.min_x+hor2.n_x]
    loar2 = np.where((mask2[0]>=cube_in.min_i) & (mask2[0]<cube_in.n_i),True,False) & np.where((mask2[1]>=cube_in.min_x) & (mask2[1]<cube_in.n_x),True,False)
    loar2h = np.where((mask[0]>=hor2.min_i) & (mask[0]<hor2.min_i+hor2.n_i),True,False) & np.where((mask[1]>=hor2.min_x) & (mask[1]<hor2.min_x+hor2.n_x),True,False)
    hdata2[loar2h] = hdata02[loar2]
    if np.nanmean(hdata2) <= np.nanmean(hdata1):
        hdata1, hdata2 = hdata2, hdata1
    
    z_step_ms = cube_in.time_step / 1000
    
    cube_time = np.arange(cube_in.data_start, cube_in.data_start  + z_step_ms * cube_in.n_samples, z_step_ms)
    grid_real, grid_not = generate_fragments(cube_in.min_i, cube_in.n_i, incr_i, cube_in.min_x, cube_in.n_x, incr_x, hdata1)
    new_zr_all = np.full((hdata1.shape[0],hdata1.shape[1]), np.nan,dtype = np.float32)
    center_x, center_y = hdata1.shape[0] // 2, hdata1.shape[1] // 2
    ind_fragments = [(i, frag, distance_to_center(frag,center_x, center_y)) for i, frag in enumerate(grid_not)]
    ind_fragments.sort(key=lambda x: x[2])
    # Выбираем индексы центральных фрагментов
    central_indices = [index for index, _, _ in ind_fragments[:num_center_fragments]]
    # Исключаем центральные фрагменты, чтобы не пересчитывать эти фрагменты повторно
    new_grid_not = [frag for i, frag in enumerate(grid_not) if i not in central_indices]
    new_grid_real = [frag for i, frag in enumerate(grid_real) if i not in central_indices]
    
    for i in central_indices:
        grid_hor_centr1 = hdata1[grid_not[i][0]:grid_not[i][0] + grid_not[i][1],grid_not[i][2]:grid_not[i][2] + grid_not[i][3]]
        grid_hor_centr2 = hdata2[grid_not[i][0]:grid_not[i][0] + grid_not[i][1],grid_not[i][2]:grid_not[i][2] + grid_not[i][3]]
        fr_minpock1, fr_maxpock1, fr_mean1, fr_median1,pocket1,raspr_count1,n,mean,M2 = compute_fragment(i,cube_in,grid_hor_centr1,grid_hor_centr2,cube_time,grid_real,pocket,n,mean,M2)
        new_fr_min.append(fr_minpock1)
        new_fr_max.append(fr_maxpock1)
        new_fr_mean.append(fr_mean1)
        new_fr_median.append(fr_median1)
    pock = auto_round((np.max(new_fr_max) - np.min(new_fr_min))/1000) # Находим резмеры кармана, шаг
    pocket = np.arange(auto_round(np.min(new_fr_min))-10000*pock, auto_round(np.max(new_fr_max))+10000*pock-1, pock)
    
    len_cube = len(new_grid_not)# Количество фрагметов большого куба
    new_count = np.zeros(len(pocket), dtype=int)
    global total_frag
    total_frag = len(new_grid_not)
    
    with ProcessPoolExecutor(max_workers=num_worker) as executor:
        futures=[]
        for k in range(len(new_grid_not)):
            grid_hor1 = hdata1[new_grid_not[k][0]:new_grid_not[k][0] + new_grid_not[k][1],new_grid_not[k][2]:new_grid_not[k][2] + new_grid_not[k][3]]
            grid_hor2 = hdata2[new_grid_not[k][0]:new_grid_not[k][0] + new_grid_not[k][1],new_grid_not[k][2]:new_grid_not[k][2] + new_grid_not[k][3]]
            f = executor.submit(compute_fragment,k,cube_in,grid_hor1,grid_hor2,cube_time,new_grid_real,pocket,n,mean,M2)
            f.add_done_callback(move_progress)
            futures.append(f)
            LOG.debug(f"Submitted: {k=}")

        # completed_frag = 0
        for f in as_completed(futures):

            try:
                fr_min1, fr_max1, fr_mean1, fr_median1,pocket1,raspr_count1,n,mean,M2 = f.result()
                
                #LOG.debug(f"Returned {z=}")
                new_fr_min.append(fr_min1)
                new_fr_max.append(fr_max1)
                new_fr_mean.append(fr_mean1)
                new_fr_median.append(fr_median1)
                if np.all(np.isnan(raspr_count1)) != True:
                    new_count += raspr_count1
                
                #LOG.debug(f"After writing to new_zr_all {z=}")
            except Exception as e:
                LOG.error(f"Exception: {e}")

        if n > 1:
            final_fr_var = M2 / n
        else:
            final_fr_var = np.nan

    return np.nanmin(new_fr_min),np.nanmax(new_fr_max),np.nanmean(new_fr_mean),np.nanmedian(new_fr_median),new_count, pocket, pock, final_fr_var
    
class statistic_horizont(di_app.DiAppSeismic3D):
    def __init__(self) -> None:
        super().__init__(in_name_par="Input Seismic3D Names", 
                out_name_par="New Name", out_names=[])

    def compute(self, f_in_tup: Tuple[np.ndarray], context: Context) -> Tuple:
        raise NotImplementedError("Shouldn't be called in this application!")

if __name__ == "__main__":
    LOG.debug(f"Starting job ExampleHor1")
    tm_start = time.time()
    job = statistic_horizont()
    cube_in = job.open_input_dataset()
    num_worker = job.description["num_worker"]
    global  incr_i, incr_x
    
    incr_i = job.description["chank_size"]
    incr_x = job.description["chank_size"]
    hor_name1 = job.description["Horizon"][0]
    hor_name2 = job.description["Horizon"][1]
    #type_interpolation = job.description["interpolation"]
    hor1 = job.session.get_horizon_3d(cube_in.geometry_name, hor_name1)
    hor2 = job.session.get_horizon_3d(cube_in.geometry_name, hor_name2)
    
    new_min,new_max,new_mean,new_median, raspr, pocket_value, step_pock,fr_var = compute_slice(cube_in, hor1, hor2,num_worker)
    ind_non_zero = np.where(raspr !=0)[0]
    raspr_non_zero = raspr[ind_non_zero[0]:ind_non_zero[-1]+1]
    value_non_zero = pocket_value[ind_non_zero[0]:ind_non_zero[-1]+1]
    LOG.info(f"{new_min=} {new_max=} {new_mean=} {new_median=} {fr_var=}")
    LOG.info(f"raspr")
    stat = {
            "data_max": float(new_max),
            "data_min": float(new_min),
            "data_mean": float(new_mean),
            "data_median": float(new_median),
            "data_var": None,
            "additional_data": {"raspr": [int(i) for i in raspr_non_zero],
                                    "first_value": value_non_zero[0],
                                    "step_pock": step_pock,
                                    "fr_var": fr_var
                                },
        }
    cube_in.save_statistics_for_horizons(hor_name1, hor_name2, stat)
    LOG.info(f"Processing time (s): {time.time() - tm_start}")