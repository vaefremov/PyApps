from typing import Optional, Tuple
import logging
from scipy.interpolate import CubicSpline,interp1d,Akima1DInterpolator
from di_lib import di_app
from di_lib.di_app import Context
import time

import numpy as np
from scipy.fft import rfftn
from scipy import signal

logging.basicConfig(level=logging.DEBUG)
LOG = logging.getLogger(__name__)
MAXFLOAT = float(np.finfo(np.float32).max)

class Butter (di_app.DiAppSeismic3D2D):
    def __init__(self) -> None:
        super().__init__(in_name_par="Input Seismic3D Names", 
                         in_line_geometries_par="Seismic2DGeometries", in_line_names_par="Input Seismic2D Names",
                out_name_par="New Name", out_names=[])
        
        # Input datasets names are converted to the agreed upon format 
        # (the CR character in  "geometry\nname\nname2" replaced by "/"", geometry name omitted)
        self.lowFreq = self.description["lowFreq"]
        self.step = self.description["step"] # input step is in ms, re-calculating to us
        self.kol_step = self.description["kol_step"]
        out_names=[]
        for n in range(self.lowFreq,self.lowFreq +self.step *self.kol_step,self.step ):
            out_names.append('spec_decomp_'+str(n)+'_'+str(n + self.step)+'_'+str(self.step))
        self.out_names = out_names

       
    def compute(self, f_in_tup: Tuple[np.ndarray], context: Context) -> Tuple:
        tm_start = time.time()
        LOG.info(f"Computing {[f_in.shape for f_in in f_in_tup]}")
        LOG.info(f"Context: {context}")
        z_step = context.out_cube_params["z_step"]
        f_in= np.where((f_in_tup[0]>= 0.1*MAXFLOAT) | (f_in_tup[0]== np.inf), np.nan, f_in_tup[0])
        fs = 1e6/z_step
        f_out=[]
        for n in range(self.lowFreq,self.lowFreq +self.step *self.kol_step,self.step ):
            b, a = signal.butter(5, [n, n + self.step], fs=fs, btype='band')
            result = (signal.filtfilt(b, a, f_in)).astype('float32')
            np.nan_to_num(result, nan=MAXFLOAT, copy=False)
            f_out.append(result)
                
        LOG.info(f"Processing time for fragment (s): {time.time() - tm_start}")

        return tuple(i for i in f_out if i is not None)

if __name__ == "__main__":
    LOG.debug(f"Starting job")
    job = Butter()
    res_final = job.run()
    LOG.info(f"{res_final}")
