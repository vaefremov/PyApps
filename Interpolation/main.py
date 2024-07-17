from typing import Optional, Tuple
import logging
import numpy as np
from scipy.interpolate import CubicSpline,interp1d
from di_lib import di_app

logging.basicConfig(level=logging.DEBUG)
LOG = logging.getLogger(__name__)
MAXFLOAT = float(np.finfo(np.float32).max)

class InterpolationZ (di_app.DiAppSeismic3D2D):
    def __init__(self) -> None:
        super().__init__(in_name_par="Input Seismic3D Name", in_line_names_par="Input Seismic2D Names", 
                out_name_par="New Name", out_names=["Interpolation"])
        
        # Input datasets names are converted to the agreed upon format 
        # (the CR character in  "geometry\nname\nname2" replaced by "/"", geometry name omitted)
        self.type_interpolation = self.description["interpolation"]
        self.new_step = self.description["step"] * 1000.0 # input step is in ms, re-calculating to us
       
    def compute(self, f_in_tup: Tuple[np.ndarray]) -> Tuple:
        LOG.info(f"Computing {[f_in.shape for f_in in f_in_tup]}")
        if (f_in_tup[0]>= 0.1*MAXFLOAT).all() or (f_in_tup[0] == np.inf).all():
            LOG.info("***EMPTY***")
            np.nan_to_num(f_in_tup[0], inf=MAXFLOAT, copy=False)
            return (f_in_tup[0],)
        
        else:
            new_nz = self.output_cubes_parameters["nz"]
            f_in= np.where((f_in_tup[0]>= 0.1*MAXFLOAT) | (f_in_tup[0]== np.inf), np.nan, f_in_tup[0])
            z = np.linspace(0,f_in_tup[0].shape[-1],f_in_tup[0].shape[-1])
            zs = np.linspace(0,f_in_tup[0].shape[-1],new_nz)
            f_out = np.empty((f_in.shape[0], f_in.shape[1], new_nz), dtype=f_in.dtype)
            if self.type_interpolation =="linear":
                try:
                    f_out = interp1d(z,f_in,axis=-1)(zs)
                except:
                    LOG.info("***MemoryError: Linear interpolation***")
            elif self.type_interpolation =="cubic spline":
                try:                 
                    f_out = CubicSpline(z,f_in,axis=-1)(zs)
                except:
                    LOG.info("***MemoryError: Сubic interpolation***")
            else:
                LOG.error(f"Unsupported interpolation type: {self.type_interpolation}")
                raise RuntimeError(f"Unsupported interpolation type: {self.type_interpolation}")
            np.nan_to_num(f_out, nan=MAXFLOAT, copy=False)
            return (f_out,)

if __name__ == "__main__":
    LOG.debug(f"Starting job")
    job = InterpolationZ()
    res_final = job.run()
    LOG.info(f"{res_final}")
