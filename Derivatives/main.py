import typing
from typing import Optional, Tuple

import logging
from multiprocessing import Pool
import time

from di_lib import di_app
from di_lib.di_app import Context

import numpy as np

MAXFLOAT = float(np.finfo(np.float32).max)

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)

def compute_derivatives(fr, dt, laplacian2d, derivative_Z):
    h_laplacian2d, h_derivative_Z = None, None
    if laplacian2d:  
        h_laplacian2d = np.full((fr.shape), np.nan, dtype = np.float32) 
        if fr is not None and len(fr.shape) == 3:
            h_laplacian2d[1:-1,1:-1,:]  = np.diff(fr, n=2, axis=0)[:,1:-1,:] + np.diff(fr, n=2, axis=1)[1:-1,:,:]
        elif fr is not None and len(fr.shape) == 2:
            h_laplacian2d[1:-1,:]  = np.diff(fr, n=2, axis=0)
        h_laplacian2d = h_laplacian2d.astype('float32')
        np.nan_to_num(h_laplacian2d, nan=MAXFLOAT, copy=False)

    if derivative_Z:
        h_derivative_Z = np.full((fr.shape), np.nan, dtype = np.float32)
        if fr is not None and len(fr.shape) == 3:
            h_derivative_Z[:,:,1:] = np.diff(fr, n=1, axis=-1) / dt
        elif fr is not None and len(fr.shape) == 2:
            h_derivative_Z[:,1:] = np.diff(fr, n=1, axis=-1) / dt
        h_derivative_Z = h_derivative_Z.astype('float32')
        np.nan_to_num(h_derivative_Z, nan=MAXFLOAT, copy=False)
    return h_laplacian2d  if laplacian2d else None, h_derivative_Z if derivative_Z else None

class Derivative(di_app.DiAppSeismic3D2D):
    def __init__(self) -> None:
        super().__init__(in_name_par="Input Seismic3D Names", 
                    in_line_geometries_par="Seismic2DGeometries", in_line_names_par="Input Seismic2D Names",
                out_name_par="New Name", out_names=[])
        
        #self.border_correction = self.description["border_correction"]
        self.laplacian2d = self.description.get("Laplacian2d", True)
        self.first_derivative_Z = self.description.get("first_derivative_Z", True)
        out_names = ["Laplacian2d", "first derivative Z"]
        self.out_flags = [self.laplacian2d , self.first_derivative_Z]
        self.out_names = [i[0] for i in zip(out_names, self.out_flags) if i[1]]

    def compute(self, f_in_tup: Tuple[np.ndarray], context: Context) -> Tuple:
        f_in = f_in_tup[0]
        f_in = np.where((f_in>= 0.1*MAXFLOAT) | (f_in== np.inf), np.nan, f_in)
        time_step = context.out_cube_params["z_step"] if context.out_cube_params else context.out_line_params["z_step"]
        if time_step is not None:
            time_step_sec = time_step/1e6
        else:
            time_step_sec = None
        f_out = compute_derivatives(f_in , time_step_sec, *self.out_flags)
        return tuple(i for i in f_out if i is not None)

if __name__ == "__main__":
    LOG.debug(f"Starting job")
    job = Derivative()
    final_result = job.run()
    LOG.info(f"{final_result}")
