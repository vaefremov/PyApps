from typing import Optional, Tuple
import logging
import numpy as np
from scipy.interpolate import CubicSpline,interp1d
from di_lib import di_app
from di_lib.di_app import Context

logging.basicConfig(level=logging.DEBUG)
LOG = logging.getLogger(__name__)
MAXFLOAT = float(np.finfo(np.float32).max)

def linear_interpolate(y, z, zs):
    good_idx = np.where( np.isfinite(y) )
    try :
        y_out = interp1d(z[good_idx], y[good_idx], axis=-1, bounds_error=False )(zs)
        return y_out
    except:
        return np.full(zs.shape[-1],np.nan)
            
def cubic_interpolate(y ,z, zs):
    good_idx = np.where( np.isfinite(y) )
    try:
        y_out = CubicSpline(z[good_idx], y[good_idx], axis=-1, extrapolate=False )(zs)
        return y_out
    except:
        return np.full(zs.shape[-1],np.nan)

class InterpolationZ (di_app.DiAppSeismic3D2D):
    def __init__(self) -> None:
        super().__init__(in_name_par="Input Seismic3D Names", in_line_names_par="Input Seismic2D Names", 
                out_name_par="New Name", out_names=["Interpolation"])
        
        # Input datasets names are converted to the agreed upon format 
        # (the CR character in  "geometry\nname\nname2" replaced by "/"", geometry name omitted)
        self.type_interpolation = self.description["interpolation"]
        self.new_step = self.description["step"] * 1000.0 # input step is in ms, re-calculating to us
        self.out_data_params["z_step"] = self.new_step
       
    def compute(self, f_in_tup: Tuple[np.ndarray], context: Context) -> Tuple:
        LOG.info(f"Computing {[f_in.shape for f_in in f_in_tup]}")
        new_nz = context.out_cube_params["nz"]
        f_in= np.where((f_in_tup[0]>= 0.1*MAXFLOAT) | (f_in_tup[0]== np.inf), np.nan, f_in_tup[0])
        z = np.linspace(0,f_in_tup[0].shape[-1],f_in_tup[0].shape[-1], dtype=f_in_tup[0].dtype)
        zs = np.linspace(0,f_in_tup[0].shape[-1],new_nz, dtype=f_in_tup[0].dtype)
              
        if self.type_interpolation == "linear":
            f_out = np.apply_along_axis(linear_interpolate, -1, f_in, z, zs)
        elif self.type_interpolation == "cubic spline":              
            f_out = np.apply_along_axis(cubic_interpolate, -1, f_in, z, zs)
            f_out  = f_out.astype('float32')
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
