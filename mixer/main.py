from typing import Optional, Tuple

import logging
import numpy as np
import time
import copy

from di_lib import di_app
from di_lib.di_app import Context

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)

MAXFLOAT = float(np.finfo(np.float32).max)

def vec_corrcoef(X, Y, axis=1):
    Xm = X - np.mean(X, axis=axis, keepdims=True)
    Ym = Y - np.mean(Y, axis=axis, keepdims=True)
    n = np.mean(Xm * Ym, axis=axis)
    d = np.std(Xm ,axis=axis)*np.std(Ym,axis=axis)
    return n / d

def corelater(traces,shift,window,p,indC,idx,win_c,att_vec):
    x    = traces[indC[0],indC[1],:]
    
    pidx = np.where((p[:,0]>=0) & (p[:,1]>=0) & (p[:,0]<traces.shape[0]) & (p[:,1]<traces.shape[1]))
    p = p[pidx]
    data = traces[p[:,0],p[:,1],:]
    
    att_vec = att_vec[pidx]
    not_nan_idx = ~np.isnan(data).all(axis=1)
    att_vec = att_vec[not_nan_idx]
    data = data[not_nan_idx]
    if shift == 0:
        mix = x + np.sum((data * att_vec[:,None]),axis=0)
    else:
        x_m    = np.lib.stride_tricks.sliding_window_view(x, axis=0, window_shape = 2 * window + 1)
        data_m = np.lib.stride_tricks.sliding_window_view(data, axis=1, window_shape = 2 * window + 1)
        cor = np.zeros((2 * shift + 1,data.shape[0],x.shape[0] - 2 * shift - 2 * window - 1 ))
        
        for j in range(-shift, shift + 1):
            cor[j,:,:] = vec_corrcoef(x_m[None,shift:x_m.shape[0] - shift - 1,:], data_m[:,j + shift:data_m.shape[1] - shift + j - 1,:], axis=2)
        cor_idx = np.argmax(cor, axis=0)
        all_idx = idx + cor_idx 
        # Выбор соответствующих сигналов из data
        result = data[np.arange(data.shape[0])[:, None], all_idx]
        mix = x[win_c + shift: x.shape[0] - window-shift] + np.sum((result * att_vec[:,None]),axis=0)
    return mix /(1.+np.sum(att_vec))

def nokta(c, frm, halfwin_traces, type_neighbors):
    if frm == '3d':
        if type_neighbors == 'square':
            idx = np.zeros(((2*halfwin_traces+1)**2-1,2),dtype=int)
            for m in range(halfwin_traces):
                for j in range(0,2*(m+1)):
                    idx[(2*m+1)**2-1 + j,:]          = [c[0] - m-1, c[1] - m - 1 + j] #для первой грани
                    idx[(2*m+1)**2-1 + 2*(m+1) + j,:]  = [c[0] - m - 1 + j, c[1] + m + 1] #для второй грани
                    idx[(2*m+1)**2-1 + 4*(m+1) + j,:] = [c[0] + m+1, c[1] + m + 1 - j] #для третьей грани
                    idx[(2*m+1)**2-1 + 6*(m+1) + j,:] = [c[0] + m+1 - j, c[1] - m - 1] #для четвертой грани 
        elif type_neighbors == 'cross':
            idx = np.zeros((halfwin_traces*4,2),dtype=int)
            for m in range(halfwin_traces):
                idx[m*4,:]     = [c[0] - m-1, c[1]] 
                idx[m*4 + 1,:] = [c[0], c[1] + m + 1] 
                idx[m*4 + 2,:] = [c[0] + m + 1, c[1]] 
                idx[m*4 + 3,:] = [c[0], c[1] - m - 1] 
    else:
        idx = np.zeros((halfwin_traces*2),dtype=int)
        for m in range(halfwin_traces):
            idx[2*m] = c - m - 1
            idx[2*m + 1] = c + m + 1
    return idx

class Mixer(di_app.DiAppSeismic3D2D):
    def __init__(self) -> None:
        super().__init__(in_name_par="Input Seismic3D Names", 
                         in_line_geometries_par="Seismic2DGeometries", in_line_names_par="Input Seismic2D Names",
                out_name_par="New Name", out_names=["mix"])
        
        self.shift      = self.description["shift"]
        self.halfwin_traces = self.description["halfwin_traces"]
        self.attenuations   = self.description["attenuations"]
        self.type_neighbors = self.description["type"]
        self.window     = 5
        self.win_c = self.window +1
        self.window = 0 if self.shift == 0 else  self.window
        self.win_c = 0 if self.shift == 0 else  self.win_c
    def compute(self, f_in_tup: Tuple[np.ndarray], context: Context) -> Tuple:
        tm_start = time.time()
        if self.type_neighbors == "square":
            att_vec = np.zeros((self.halfwin_traces * 2 + 1)**2 - 1)
            for i in range(self.halfwin_traces):
                att_vec[(i * 2 + 1) ** 2 - 1 : ((i + 1) * 2 + 1) ** 2 - 1] = self.attenuations ** (i + 1) 

        elif self.type_neighbors == "cross":
            att_vec = np.zeros(self.halfwin_traces * 4)
            for i in range(self.halfwin_traces):
                att_vec[i * 4 : (i + 1) * 4] = self.attenuations **(i + 1) 

        f_in = copy.deepcopy(f_in_tup[0])
        f_in = np.where((f_in>= 0.1*MAXFLOAT) | (f_in== np.inf), np.nan, f_in)
        idx = np.arange(self.window+self.shift+1, f_in.shape[-1] - self.window-self.shift)
        if len(f_in.shape) == 3:
            frm = '3d'
            newTraces = np.full(f_in.shape, np.nan, dtype=np.float32)
            for i in range(0,f_in.shape[0]):
                for j in range(0,f_in.shape[1]):
                    if np.isinf(f_in[i,j,:]).all() == True:
                        continue
                    else:
                        indC = [i, j]
                        indAll = nokta(indC, frm, self.halfwin_traces, self.type_neighbors)
                        mix_sig = corelater(f_in, self.shift, self.window, indAll, indC, idx,self.win_c, att_vec)
                        newTraces[indC[0],indC[1], self.win_c + self.shift: f_in.shape[2] - self.window - self.shift] = mix_sig
        elif len(f_in.shape) == 2:
            frm = '2d'
            att_vec = np.zeros(self.halfwin_traces * 2)
            for i in range(self.halfwin_traces):
                att_vec[i * 2 : (i + 1) * 2] = self.attenuations ** (i + 1) 
            #newTraces = f_in
            newTraces = np.full(f_in.shape, np.nan, dtype=np.float32)
            for i in range(0, f_in.shape[0]):
                if np.isinf(f_in[i,:]).all() == True:
                    continue
                else:
                    indC = i
                    indAll = nokta(indC, frm, self.halfwin_traces, self.type_neighbors)
                    mix_sig = corelater(f_in, self.shift, self.window, indAll, indC, idx,self.win_c, att_vec)
                    newTraces[indC[0],indC[1], self.win_c + self.shift: f_in.shape[2] - self.window - self.shift] = mix_sig
        else:
            raise ValueError(f"Unsupported input shape: {f_in.shape}")
        newTraces = newTraces.astype('float32')
        np.nan_to_num(newTraces, nan=MAXFLOAT, copy=False)
        LOG.info(f"Processing time for fragment (s): {time.time() - tm_start}")
        return (newTraces,)

if __name__ == "__main__":
    LOG.debug(f"Starting job")
    job = Mixer()
    res_final = job.run()
    LOG.info(f"{res_final}")
