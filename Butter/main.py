from typing import Optional, Tuple
import logging
from scipy.interpolate import CubicSpline,interp1d,Akima1DInterpolator
from di_lib import di_app
from di_lib.di_app import Context
import numpy as np
from scipy.fft import rfftn
from scipy import signal

logging.basicConfig(level=logging.DEBUG)
LOG = logging.getLogger(__name__)
MAXFLOAT = float(np.finfo(np.float32).max)

class Butter (di_app.DiAppSeismic3D2D):
    def __init__(self) -> None:
        super().__init__(in_name_par="Input Seismic3D Names", in_line_names_par="Input Seismic2D Names", 
                out_name_par="New Name", out_names=["Butter"])
        
        # Input datasets names are converted to the agreed upon format 
        # (the CR character in  "geometry\nname\nname2" replaced by "/"", geometry name omitted)
        self.lowFreq = self.description["lowFreq"]
        self.step = self.description["step"] # input step is in ms, re-calculating to us
	self.kol_step = self.description["kol_step"]
        self.out_data_params["z_step"] = self.description["z_step"]
       
    def compute(self, f_in_tup: Tuple[np.ndarray], context: Context) -> Tuple:
        LOG.info(f"Computing {[f_in.shape for f_in in f_in_tup]}")
        new_nz = context.out_cube_params["nz"]
        #f_in= np.where((f_in_tup[0]>= 0.1*MAXFLOAT) | (f_in_tup[0]== np.inf), np.nan, f_in_tup[0])
        for i in range(lowFreq, lowFreq + kol_step + step + 2, step):
    		b, a = signal.butter(4, [i, i+1], fs=fs, btype='band')
    		f_out = signal.lfilter(b, a, f_in)
        np.nan_to_num(f_out, nan=MAXFLOAT, copy=False)
	LOG.info(f"Processing time for fragment (s): {time.time() - tm_start}")

        return (f_out,)

if __name__ == "__main__":
    LOG.debug(f"Starting job")
    job = Butter()
    res_final = job.run()
    LOG.info(f"{res_final}")
