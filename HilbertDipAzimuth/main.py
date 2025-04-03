import typing
from typing import Optional, Tuple

import logging
from multiprocessing import Pool
import time

from di_lib import di_app
from di_lib.di_app import Context

import numpy as np
import scipy as sc
import scipy.signal as signal

MAXFLOAT = float(np.finfo(np.float32).max)

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)

def taper_fragment(fr, border_correction: int):
    bl = np.blackman(border_correction*2+1)
    mask = np.ones(fr.shape[-1], dtype=np.float32)
    mask[:border_correction] = bl[:border_correction]
    mask[-border_correction:] = bl[border_correction+1:]
    res = np.multiply(fr, mask)
    return res

def compute_hilbert_transform(fr, min_frequency, max_frequency, radius, dt, w_amp, w_phase, w_freq, w_azimuth, w_dip, w_sweetness):
    h_amp, h_phase, h_freq, h_azimuth, h_dip, h_sweetness = None, None, None, None, None, None
    h = signal.hilbert(fr)
    h = typing.cast(np.ndarray, h)
    
    if w_amp or w_sweetness:
        h_amp = np.abs(h)
        np.nan_to_num(h_amp, nan=MAXFLOAT, copy=False)
    if w_phase or w_azimuth or w_dip:
        h_phase = np.arctan2(np.real(h), np.imag(h))
        np.nan_to_num(h_phase, nan=MAXFLOAT, copy=False)
    if w_freq or w_azimuth or w_dip or w_sweetness:
        h_freq = (1/(2*np.pi*dt)) * np.imag(np.gradient(h, axis=-1) / h)
        h_freq[h_freq < min_frequency] = min_frequency
        h_freq[h_freq > max_frequency] = max_frequency
        np.nan_to_num(h_freq, nan=MAXFLOAT, copy=False)
    if w_azimuth or w_dip:
        k = 0.5
        ph_grad = np.asarray(np.gradient(h_phase))
        ph_grad[0] = np.where(ph_grad[0] >= k * np.pi, ph_grad[0] - np.pi, ph_grad[0])
        ph_grad[0] = np.where(ph_grad[0] <= -k * np.pi, ph_grad[0] + np.pi, ph_grad[0])
        if len(fr.shape)==3:
            ph_grad[1] = np.where(ph_grad[1] >= k * np.pi, ph_grad[1] - np.pi, ph_grad[1])    
            ph_grad[1] = np.where(ph_grad[1] <= -k * np.pi, ph_grad[1] + np.pi, ph_grad[1])
            h_azimuth  = np.arctan2(ph_grad[1], ph_grad[0])
        elif len(fr.shape)==2:
            h_azimuth  = np.arctan2(0,ph_grad[0])
        np.nan_to_num(h_azimuth, nan=MAXFLOAT, copy=False)
    if w_dip:
        q = np.divide(ph_grad[0], h_freq)
        if len(fr.shape)==3:
            p = np.divide(ph_grad[1], h_freq)
            h_dip = np.sqrt(p**2 + q**2)
        elif len(fr.shape)==2:
            h_dip = np.abs(q)
        np.nan_to_num(h_dip, nan=MAXFLOAT, copy=False)
    if w_sweetness:
        h_sweetness = np.full(fr.shape, np.nan, dtype = np.float32)
        h_amp_interim = np.where((h_amp>= 0.1*MAXFLOAT) | (h_amp== np.inf), np.nan, h_amp)
        h_freq_interim = np.where((h_freq>= 0.1*MAXFLOAT) | (h_freq== np.inf), np.nan, h_freq)
        mean_h_amp = np.nanmean(np.lib.stride_tricks.sliding_window_view(h_amp_interim,  axis=-1, window_shape = 2 * radius + 1),axis=-1)
        mean_h_freq = np.nanmean(np.lib.stride_tricks.sliding_window_view(h_freq_interim, axis=-1, window_shape = 2 * radius + 1),axis=-1)
        if len(fr.shape)==3:
            h_sweetness[:,:,radius:-radius] = mean_h_amp / np.sqrt(mean_h_freq)
        elif len(fr.shape)==2:
            h_sweetness[:,radius:-radius] = mean_h_amp / np.sqrt(mean_h_freq)
        np.nan_to_num(h_sweetness, nan=MAXFLOAT, copy=False)
    return h_amp if w_amp else None, h_phase if w_phase else None, h_freq if w_freq else None, h_azimuth if w_azimuth else None,\
          h_dip if w_dip else None, h_sweetness if w_sweetness else None
    
class DipAzimuth(di_app.DiAppSeismic3D2D):
    def __init__(self) -> None:
        super().__init__(in_name_par="Input Seismic3D Names", 
                         in_line_geometries_par="Seismic2DGeometries", in_line_names_par="Input Seismic2D Names", 
                out_name_par="New Name",out_names=[])
        
        self.min_frequency = self.description["min_frequency"]
        self.max_frequency = self.description["max_frequency"]
        self.radius = self.description["radius"]
        self.border_correction = self.description["border_correction"]
        self.compute_amplitude = self.description.get("inst_amplitude", True)
        self.compute_frequency = self.description.get("inst_frequency", True)
        self.compute_phase = self.description.get("phase", True)
        self.compute_azimuth = self.description.get("azimuth", True)
        self.compute_dip = self.description.get("dip", True)
        self.compute_sweetness = self.description.get("sweetness", True)

        out_names = ["Instantaneous amplitude", "Instantaneous phase", "Instantaneous frequency", "Azimuth", "Dip","Sweetness"]
        self.out_flags = [self.compute_amplitude, self.compute_frequency, self.compute_phase, self.compute_azimuth, self.compute_dip,self.compute_sweetness]
        self.out_names = [i[0] for i in zip(out_names, self.out_flags) if i[1]]

    def compute(self, f_in_tup: Tuple[np.ndarray], context: Context) -> Tuple:
        f_in = f_in_tup[0]
        tmp_f = taper_fragment(f_in, self.border_correction)
        time_step = context.out_cube_params["z_step"] if context.out_cube_params else context.out_line_params["z_step"]
        if time_step is not None:
            time_step_sec = time_step/1e6
            radius_sample = int(self.radius/(time_step_sec*1000))
        else:
            time_step_sec = None
            radius_sample = None            
        f_out = compute_hilbert_transform(tmp_f, self.min_frequency,
                self.max_frequency, radius_sample, time_step_sec, *self.out_flags)
        return tuple(i for i in f_out if i is not None)

if __name__ == "__main__":
    LOG.debug(f"Starting job")
    job = DipAzimuth()
    final_result = job.run()
    LOG.info(f"{final_result}")
