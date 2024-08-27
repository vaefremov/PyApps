from typing import Optional, Tuple
import logging
from di_lib import di_app
from di_lib.di_app import Context
import time

import numpy as np
import math
from scipy.signal import hilbert, cwt, ricker, convolve, butter, filtfilt

logging.basicConfig(level=logging.DEBUG)
LOG = logging.getLogger(__name__)
MAXFLOAT = float(np.finfo(np.float32).max)

def f2w(f, fs):
    return fs/(math.sqrt(2.0) * np.pi * f)

def decomp_CWT(f_in, width):
    result = np.abs(hilbert(cwt(f_in,ricker,width)))
    return result

def decomp_STFT(f_in, c_exp):
    result = np.abs(convolve(f_in, c_exp, mode='same'))
    return result

class Decomposition (di_app.DiAppSeismic3D2D):
    def __init__(self) -> None:
        super().__init__(in_name_par="Input Seismic3D Names", 
                         in_line_geometries_par="Seismic2DGeometries", in_line_names_par="Input Seismic2D Names",
                out_name_par="New Name", out_names=[])
        
        # Input datasets names are converted to the agreed upon format 
        # (the CR character in  "geometry\nname\nname2" replaced by "/"", geometry name omitted)
        self.lowFreq = self.description["lowFreq"]
        self.step = self.description["step"] # input step is in ms, re-calculating to us
        self.num_steps = self.description["num_steps"]
        self.window_width = self.description["window_width"]*1e3
        self.type_decomposition = self.description["type_decomposition"]

        out_names   = []
        frequencies = []
            
        if self.type_decomposition =="BPF":
            for n in range(self.lowFreq,self.lowFreq +self.step *self.num_steps,self.step ):
                out_names.append(self.type_decomposition+'_'+str(n)+'_'+str(n + self.step)+'_'+str(self.step))
                frequencies.append([n, n + self.step])
        else:
            for n in range(self.lowFreq, self.lowFreq + self.step * (self.num_steps+1), self.step ):
                out_names.append(self.type_decomposition+'_'+str(n)+'_'+str(self.step))
                frequencies.append(n)

        self.frequencies = frequencies
        self.out_names = out_names
    
    def compute(self, f_in_tup: Tuple[np.ndarray], context: Context) -> Tuple:
        tm_start = time.time()
        LOG.info(f"Computing {[f_in.shape for f_in in f_in_tup]}")
        LOG.info(f"Context: {context}")
        z_step = context.out_cube_params["z_step"]
        f_in= np.where((f_in_tup[0]>= 0.1*MAXFLOAT) | (f_in_tup[0]== np.inf), np.nan, f_in_tup[0])
        fs = 1e6/z_step
        N = f_in.shape[-1]
        f_out =  []
        if self.type_decomposition == 'STFT' :
            npoints = np.floor(self.window_width / z_step).astype('int')
            t = np.linspace(0., self.window_width/1e6, npoints+1) 
            c_exp = [np.exp(-1.j*np.pi * 2 * f * t) for f in self.frequencies]
            for e in c_exp:
                result = (np.apply_along_axis(decomp_STFT, -1, f_in, e)).astype('float32')
                np.nan_to_num(result, nan=MAXFLOAT, copy=False)
                f_out.append(result)
        
        if self.type_decomposition == 'CWT' :
            widths=[f2w(f, fs) for f in self.frequencies]
            result = (np.apply_along_axis(decomp_CWT, -1, f_in, widths)).astype('float32')
            np.nan_to_num(result, nan=MAXFLOAT, copy=False)
            f_out=[result[:,:,i,:] for i in range(len(self.frequencies))] 

        if self.type_decomposition == 'BPF' :
            for n in self.frequencies:
                b, a = butter(5, n, fs=fs, btype='band')
                result = (np.abs(hilbert(filtfilt(b, a, f_in)))).astype('float32')
                np.nan_to_num(result, nan=MAXFLOAT, copy=False)
                f_out.append(result) 

        LOG.info(f"Processing time for fragment (s): {time.time() - tm_start}")

        return tuple(i for i in f_out if i is not None)

if __name__ == "__main__":
    LOG.debug(f"Starting job")
    job = Decomposition()
    res_final = job.run()
    LOG.info(f"{res_final}")
