from typing import Optional, Tuple

import logging
import numpy as np
import time

from di_lib import di_app
from di_lib.di_app import Context

logging.basicConfig(level=logging.DEBUG)
LOG = logging.getLogger(__name__)

MAXFLOAT = float(np.finfo(np.float32).max)


def vec_corrcoef(X, Y, axis=1):
    Xm = X - np.mean(X, axis=axis, keepdims=True)
    Ym = Y - np.mean(Y, axis=axis, keepdims=True)
    n = np.mean(Xm * Ym, axis=axis)
    d = np.std(X ,axis=axis)*np.std(Ym,axis=axis)
    #d = np.sqrt(np.sum((X - Xm)**2, axis=axis) * np.sum((Y - Ym)**2, axis=axis))
    return n / d

def corr(x,data,window,shift):
    x_m    = np.lib.stride_tricks.sliding_window_view(x, axis=0, window_shape = 2 * window + 1)
    data_m = np.lib.stride_tricks.sliding_window_view(data, axis=1, window_shape = 2 * window + 1)
    if shift == 0:
        cor = np.min(vec_corrcoef(x_m[None,:,:], data_m, axis=2),axis=0)
    else:
        cor = np.zeros((2 * shift + 1,data.shape[0],x.shape[0] - 2 * shift - 2 * window))
        for j in range(-shift,shift + 1):
            cor[j,:,:] = vec_corrcoef(x_m[None,shift:x_m.shape[0] - shift,:], data_m[:,j + shift:data_m.shape[1]-shift + j,:], axis=2)
        cor = np.min(np.max(cor,axis=0),axis=0)
    return cor

def corelater(Traces1,shift,window,p,indC,frm):
    if frm == '3d':
        a = Traces1[indC[0],indC[1],:]
        b = Traces1[p[:,0],p[:,1],:]
    else:
        a = Traces1[indC,:]
        b = Traces1[p,:]

    return corr(a,b,window,shift)

def nokta(a,frm):
    if frm == '3d':
        indx = np.zeros((8,2),dtype=int)
        indx[0,:] = [a[0]-1,a[1]-1] 
        indx[1,:] = [a[0]-1,a[1]] 
        indx[2,:] = [a[0]-1,a[1]+1]    
        indx[3,:] = [a[0],a[1]-1]
        indx[4,:] = [a[0],a[1]+1] 
        indx[5,:] = [a[0]+1,a[1]-1] 
        indx[6,:] = [a[0]+1,a[1]]
        indx[7,:] = [a[0]+1,a[1]+1]
    else:
        indx = np.zeros((2),dtype=int)
        indx[0] = a-1
        indx[1] = a+1

    return indx

class Coherence(di_app.DiAppSeismic3D2D):
    def __init__(self) -> None:
        super().__init__(in_name_par="Input Seismic3D Names", in_line_names_par="Input Seismic2D Names",
                out_name_par="New Name", out_names=["Coherence"])
        
        self.min_window = self.description["window"]
        self.max_shift = self.description["shift"]

    def compute(self, f_in_tup: Tuple[np.ndarray], context: Context) -> Tuple:
        tm_start = time.time()
        f_in = f_in_tup[0]
        if len(f_in.shape) == 3:
            frm = '3d'
            newTraces = np.zeros(shape=(f_in.shape[0], f_in.shape[1], f_in.shape[2]), dtype=np.float32)
            newTraces[:] = np.nan
            for i in range(1,f_in.shape[0]-1):
                for j in range(1,f_in.shape[1]-1):
                    if np.isinf(sum(f_in[i,j,:])) == True:
                        continue
                    else:
                        indC = [i,j]
                        indAll = nokta(indC,frm)
                        sig = corelater(f_in,self.max_shift,self.min_window,indAll,indC,frm)
                        newTraces[indC[0],indC[1],self.min_window + self.max_shift:f_in.shape[2] - self.min_window-2 * self.max_shift + self.max_shift] = sig
        elif len(f_in.shape) == 2:
            frm = '2d'
            newTraces = np.zeros(shape=(f_in.shape[0], f_in.shape[1]), dtype=np.float32)
            newTraces[:] = np.nan
            for i in range(1,f_in.shape[0]-1):
                if np.isinf(sum(f_in[i,:])) == True:
                    continue
                else:
                    indC = i
                    indAll = nokta(indC,frm)
                    sig = corelater(f_in,self.max_shift,self.min_window,indAll,indC,frm)
                    newTraces[indC,self.min_window + self.max_shift:f_in.shape[1] - self.min_window-2 * self.max_shift + self.max_shift] = sig
        else:
            raise ValueError(f"Unsupported input shape: {f_in.shape}")
        np.nan_to_num(newTraces, nan=MAXFLOAT, copy=False)
        LOG.info(f"Processing time for fragment (s): {time.time() - tm_start}")
        return (newTraces,)

if __name__ == "__main__":
    LOG.debug(f"Starting job")
    job = Coherence()
    res_final = job.run()
    LOG.info(f"{res_final}")
