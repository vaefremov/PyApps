import logging
import numpy as np
from typing import Any, List, Optional, Tuple, Dict, Union, cast
from scipy.interpolate import CubicSpline,interp1d,Akima1DInterpolator
from collections import namedtuple, Counter
import numexpr as ne
import os
import argparse
import sys
import os
import multiprocessing
import signal
import functools
import time
import traceback
import abc
#from session import DISession
from typing import Any, List, Optional, Tuple, Dict, Union, cast
from di_lib import di_app
from di_lib.di_app import Context
from concurrent.futures import ProcessPoolExecutor, wait, Future, as_completed
from di_lib.attribute import DIAttribute2D, DIHorizon3D, DIHorizon3DWriter

MAXFLOAT = float(np.finfo(np.float32).max)
logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
class Frag(namedtuple("Frag", "no_i span_i no_x span_x")):
    __slots__ = ()
# LOG = logging.getLogger(__name__)
# MAX_TRIES = 10

# class JobDescription(namedtuple("JobDescription", ["job_id", "project_id", "token", "server_url"])):
#     __slots__ = ()

# class Frag(namedtuple("Frag", "no_i span_i no_x span_x")):
#     __slots__ = ()

# class Context(namedtuple("Context", "in_cube_params in_line_params out_cube_params out_line_params")):
#     __slots__ = ()

class ProcessCParams(namedtuple("ProcessCParams", "c_in c_out frag")):
    __slots__ = ()

class ProcessLParams(namedtuple("ProcessLParams", "nm, p_in, p_out")):
    __slots__ = ()
    
def children_sigterm_handler(signum, frame):
    LOG.debug(f"Signal caught in child {signum}")
    sys.exit(signum)

def main_sigint_handler_with_exec(executor: ProcessPoolExecutor, signum, frame):
    LOG.info(f"Caught signal {signum} in main process")
    if executor:
        executor.shutdown(wait=False)
        LOG.debug(f"Executor shutdown")
    for child in multiprocessing.active_children():
        print(f"Killing {child.pid}")
        child.kill()
    print("after killing subprocesses")
    os._exit(signum)
def init_process():
    LOG.debug(f"Signals reset in child process {os.getpid()}")
    signal.signal(signal.SIGTERM, children_sigterm_handler)
    signal.signal(signal.SIGINT, children_sigterm_handler)
def enlarge_fragment(frag: Frag, marg: int) -> Frag:
    no_i_f = frag.no_i - marg
    span_i_f = frag.span_i + 2*marg
    if no_i_f < 0:
        span_i_f = frag.span_i + (marg + no_i_f) + marg
        no_i_f = 0
    no_x_f = frag.no_x - marg
    span_x_f = frag.span_x + 2*marg
    if no_x_f < 0:
        span_x_f = frag.span_x + (marg + no_x_f) + marg
        no_x_f = 0
    return Frag(no_i_f, span_i_f, no_x_f, span_x_f)
def linear_interpolate(y, z, zs):
    good_idx = np.where( np.isfinite(y) )
    try :
        y_out = interp1d(z[good_idx], y[good_idx], axis=-1, bounds_error=False )(zs)
        return y_out
    except:
        return np.full(zs.shape[-1],np.nan)
    
def generate_fragments_grid_incr(min_i, n_i, inc_i, min_x, n_x, inc_x):
    """Generate grid with constant increments over inlines/xlines.
    """
    res_real = [(i, j, min(n_i-1, i+inc_i-1), min(n_x-1, j+inc_x-1)) for i in range(min_i, n_i-1, inc_i) for j in range(min_x, n_x-1, inc_x)]
    return res_real
    
@abc.abstractmethod
def generate_process_arguments(self) -> Dict[int, Union[ProcessCParams, ProcessLParams]]:
    return {}

class DiAppSeismic3DMultipleCustom(di_app.DiApp):
    """This class launches calculations on multiple input data belonging to the same geometry.
    """
    def __init__(self, in_name_par: str,in_name_attr:str,in_name_hor:str, out_name_par: str, out_names: List[str],result_type:str) -> None:
        super().__init__()
        self.in_name_par = in_name_par
        self.out_names = out_names
        self.out_name_par = out_name_par
        self.in_name_attr = in_name_attr
        self.in_name_hor = in_name_hor
        self.result_type = result_type
        #self.cube_in: Optional[DISeismicCube] = None

    def open_input_datasets(self) -> list:
        #attn
        cube_names = self.description[self.in_name_par]
        attr_names = self.description[self.in_name_attr]
        hor_names  = self.description[self.in_name_hor]  
        # Check if all data belong to the same geometry
        #Отключил проверку ,так как в интерфейсе калькулятора уже предусметрено 

        cubes = [self.session.get_cube(cn['geometry_name'], cn['name'], cn['name2']) for cn in cube_names]
        #attrs = [self.session.get_attribute_2d_writer(cn[-1],cn[0], "/".join(cn[1:-1])) for cn in attr_names]
        attrs = [self.session.get_attribute_2d_writer(cn['geometry_name'], cn['name'], cn['name2']) for cn in attr_names]
        hors = [self.session.get_horizon_3d(cn['geometry_name'], cn['name']) for cn in hor_names]
        min_i = []
        n_i = []
        min_x = []
        n_x = []
        if self.output_data_type == "seismic_3d":
            # Check that all input cubes have the same time-axis parameters
            c0 = cubes[0]
            cubes_time_steps = []
            data_starts = []
            cubes_n_samples = []
            max_time = []

            for c in cubes:
                cubes_time_steps.append(c.time_step)
                data_starts.append(c.data_start)
                cubes_n_samples.append(c.n_samples)
                max_time.append(c.data_start+c.time_step*c.n_samples)

                min_i.append(c.min_i)
                n_i.append(c.n_i)
                min_x.append(c.min_x)
                n_x.append(c.n_x)
                if c.domain != c0.domain:
                    raise RuntimeError("Time axis parameters of the input cubes do not coincide")
            
            self.z_step  = min(cubes_time_steps)
            self.z_start = max(data_starts)
            self.n_samples = int((min(max_time)-self.z_start)//self.z_step)
        elif attr_names!=[]:
            cubes = []
            for c in attrs:
                min_i.append(c.min_i)
                n_i.append(c.n_i)
                min_x.append(c.min_x)
                n_x.append(c.n_x)

        elif hor_names!=[]: 
            cubes =[]
            for c in hors: 
                min_i.append(c.min_i)
                n_i.append(c.n_i)
                min_x.append(c.min_x)
                n_x.append(c.n_x)

        self.n_i = min(n_i) 
        self.n_x = min(n_x)
        self.min_i = max(min_i)
        self.min_x = max(min_x)
        dates = cubes + attrs + hors
        LOG.debug(f"{dates=}")
        return dates

    def create_output_datasets(self):

        #if type(self.data_in[0]).__name__ == "DISeismicCube":
        if self.output_data_type == "seismic_3d":
            cube_in = self.data_in[0]
            res = []
            
            for result_name in self.out_names:
                name = self.description[self.out_name_par]
                name2 = f"{result_name}"
                c_out = self.session.create_cube_writer_in_geometry(cube_in.geometry_name, name, name2, z_step = self.z_step, z_start = self.z_start, nz = self.n_samples,max_inline=self.n_i-1,
                                                                    max_xline = self.n_x-1, min_inline = self.min_i,min_xline = self.min_x,**self.out_data_params)# здесь должен быть правильный z_step,z_start
                res.append(c_out)
        elif self.output_data_type == "horizonAttributes_3d":
            hor = self.data_in[-1]
            res = []

            for result_name in self.out_names:
                
                name = self.description[self.out_name_par]
                name2 = f"{result_name}"
                c_out = self.session.create_attribute_2d_writer_as_other(hor, name, name2, domain="D", copy_horizon_data=True)
                res.append(c_out)

        elif self.output_data_type == "horizons_3d":
            hor = self.data_in[-1]
            res = []
            
            for result_name in self.out_names:
                name = self.description[self.out_name_par]
                name2 = f"{result_name}"
                #if self.cubes_in!={}:
                c_out = self.session.create_horizon_3d_writer_as_other(hor, name)
                res.append(c_out)
        return res
    def generate_fragments_grid_incr(self, incr_i, incr_x):
        tmp = generate_fragments_grid_incr(self.min_i, self.n_i, incr_i, self.min_x, self.n_x, incr_x)
        return [(i[0], i[2]-i[0]+1, i[1], i[3]-i[1]+1) for i in tmp]
    
    def generate_optimal_grid(self):
        """Currently the first cube in the input dataset is used to generate fragmens grig.
        """
        incr_i = 64
        incr_x = 64
        #c = self.data_in[0]
        # найти места пересечения
        if self.output_data_type == "horizonAttributes_3d":
            
            #grid = self.generate_fragments_grid_incr(incr_i=self.n_i, incr_x =self.n_x)
            grid = [(self.min_i, self.n_i, self.min_x, self.n_x)]
        else:
            grid = self.generate_fragments_grid_incr(incr_i, incr_x)
        self.grid = grid
        return grid

    def generate_process_arguments(self) -> Dict[int, ProcessCParams]:
        data_in = self.open_input_datasets()
        self.data_in = data_in
        #найти места пересечения
        grid = self.generate_optimal_grid()
        #grid = generate_optimal_grid(min_i, n_i, incr_i, min_x, n_x, incr_x)

        w_out = self.create_output_datasets()
        
        LOG.debug(f"{grid=}")

        enum_grid = enumerate(grid)
        self.total_frags = len(grid)
        run_args = {i[0]: ProcessCParams(data_in, w_out, i[1]) for i in enum_grid}
        return run_args  
    def get_fragment_2D(self,c_in: Optional[str],  inline_no: int, inline_count: int, xline_no: int, xline_count: int) -> Optional[np.ndarray]:
        arg = c_in.get_data()
        return arg[inline_no - self.min_i:inline_no + inline_count - self.min_i + 1,xline_no - self.min_x:xline_no + xline_count - self.min_x + 1] 

    def process_data(self, task_id, params: ProcessCParams) -> Tuple[int, str]:
        def output_frag_if_not_none(w_out, f_out, f_coords):
            if w_out:
                if self.output_data_type == "seismic_3d":
                    w_out.write_fragment(f_coords[0], f_coords[2], f_out)
                elif self.output_data_type == "horizonAttributes_3d":
                    w_out.write_data(f_out)
                else:
                    w_out.write_data_fragment(f_coords[0], f_coords[2], f_out)
        #tmp_f = tuple([c.get_fragment_z(*params.frag,z_no=2,z_count=30) for c in params.c_in])
        #Если приходит куб, использовать готовую функцию чтения по фрагментам для класса куб, иначе get_fragment_2D
        #if self.output_data_type == "horizonAttributes_3d":
        #    tmp_f = tuple([self.get_fragment_2D(c,*params.frag) for c in params.c_in])
        #else:    
        tmp_f = tuple([c.get_fragment(*params.frag) if type(c).__name__ == "DISeismicCube"  else self.get_fragment_2D(c,*params.frag) for c in params.c_in])

        for f in tmp_f:
            if f is None:
                LOG.debug(f"Skipped: {task_id} {params.frag}")
                return task_id, "SKIP"
        out_cube_params = params.c_out[0]._get_info() if len(params.c_out) else None
        context = Context(in_cube_params=params.c_in[0]._get_info(), in_line_params=None, out_cube_params=out_cube_params, out_line_params=None)
        context.in_cube_params["chunk"] = params.frag
        f_out = ()
        f_out = self.compute(tmp_f,params.c_in, context=context)
        if di_app.DiApp.wrong_output_formats(tmp_f, f_out):
            raise RuntimeError(f"Wrong output array format: shape or dtype do not coinside with input")
        for w,f in zip(params.c_out, f_out):
            output_frag_if_not_none(w, f, params.frag)
        LOG.debug(f"Processed {task_id} {params.frag}")
        return task_id, "OK"
        
    def process_fragment(self, task_id, params: Union[ProcessCParams, ProcessLParams]) -> Tuple[int, str]:
        try:
            if issubclass(type(params), ProcessCParams):
                return self.process_data(task_id, cast(ProcessCParams, params))
            else:
                raise RuntimeError(f"Unexpected type of argument {type(params)}")
        except Exception as ex:
            traceback.print_exc()
            ex.args = (task_id,) + ex.args
            raise ex

class SeismicCalculation(DiAppSeismic3DMultipleCustom):
    def __init__(self) -> None:
        super().__init__(in_name_par="seismic_3d",in_name_attr='horizonAttributes_3d',in_name_hor='horizons_3d',
                out_name_par="result_name", out_names=["formula"],result_type = 'result_type')
        # Input datasets names are converted to the agreed upon format 
        # (the CR character in  "geometry\nname\nname2" replaced by "/"", geometry name omitted)
        #cube_names_for_formula = ["/".join(nn.split("\n")[1:]) for nn in self.description[self.in_name_par]]
        cube_names_for_formula = self.description[self.in_name_par]
        hor_names_for_formula = self.description[self.in_name_hor]
        attr_names_for_formula = self.description[self.in_name_attr]
        self.formula = self.description["formula"]
        self.output_data_type = self.description['result_type']
        # if 'seismic_3d' in self.description["formula"]:
        #     self.output_data_type = "DISeismicCube"
        # else:
        #     if 'horizons_3d' in self.description["formula"] and 'horizonAttributes_3d' not in self.description["formula"]:
        #         self.output_data_type = "DIHorizon3D"
        #     else:
        #         self.output_data_type = "DIAttribute2D"
        # Converting formula to the format that can be used in the compute() method context
        # Datasets names replaced with references to the corresponding fragment in f_in_tup
        # argument of compute(). Replacement are applied in reverse order of name lengths, most
        # long names replaced first.
        name_count=0
        if self.output_data_type == "seismic_3d":
            #cube_names_for_formula = [i.split("/") for i in cube_names_for_formula]   
            #cube_names_for_formula = ["/".join(i[1:]) for i in cube_names_for_formula] 
            for num, nm in reversed(sorted(enumerate(cube_names_for_formula), key=lambda x: len(x[1]["name"]))):
                self.formula = self.formula.replace("<seismic_3d>"+nm["name"]+'/'+nm["name2"], f"variable{num}")
            name_count+=num
        elif 'horizonAttributes_3d' in self.description["formula"]:
            for num, nm in reversed(sorted(enumerate(attr_names_for_formula), key=lambda x: len(x[1]["name"]))):
                self.formula = self.formula.replace("<horizonAttributes_3d>"+nm["name"]+'/'+nm["name2"], f"variable{name_count+num}")
            name_count+=num
        else:
            for num, nm in reversed(sorted(enumerate(hor_names_for_formula), key=lambda x: len(x[1]["name"]))):
                if 'horizons_3d' in self.description["formula"]:
                    self.formula = self.formula.replace("<horizons_3d>"+nm["name"], f"variable{name_count+num}")
            name_count+=num
        self.formula = self.formula.lower()
        self.formula = self.formula.strip()
         
        
        LOG.info(f"\n ***FORMULA*** \nOriginal formula: {self.description['formula']} \nFinal formula: {self.formula}")
    def compute(self, f_in_tup: Tuple[np.ndarray],data_in, context: Context) -> Tuple:
        LOG.debug(f"Computing {[f_in.shape for f_in in f_in_tup]}")

        if (f_in_tup[0]>= 0.1*MAXFLOAT).all() :
            LOG.debug("***EMPTY***")
            return (f_in_tup[0],)
        elif (f_in_tup[0] == np.inf).all():
            LOG.debug("***EMPTY***")
            np.nan_to_num(f_in_tup[0], inf=MAXFLOAT, copy=False) 
            return (f_in_tup[0],)
        
        else:
            variable = list(f_in_tup)
            for i, v in enumerate(f_in_tup):             
                variable[i] = np.where((variable[i] >= 0.1*MAXFLOAT) | (variable[i] == np.inf), np.nan, variable[i])
                if len(variable[i].shape)==3:
                    if data_in[i].time_step != self.z_step:
                        z = np.linspace(0,f_in_tup[i].shape[-1],f_in_tup[i].shape[-1], dtype=f_in_tup[i].dtype)
                        zs = np.linspace(0,f_in_tup[i].shape[-1],self.z_step, dtype=f_in_tup[i].dtype)
                        variable[i] = np.apply_along_axis(linear_interpolate, -1, f_in_tup[i], z, zs)
                        variable[i] = variable[i][:,:,int(1E3*(self.z_start-data_in[i].data_start)/self.z_step):int(1E3*(self.z_start-data_in[i].data_start)/self.z_step+self.n_samples)]
                    else:
                        variable[i] = variable[i][:,:,int(1E3*(self.z_start-data_in[i].data_start)/self.z_step):int(1E3*(self.z_start-data_in[i].data_start)/self.z_step+self.n_samples)]
                elif len(variable[i].shape)!=3 and  "seismic_3d" in self.description["formula"] :
                    variable[i] = variable[i][:,:,None]  
                globals() ["variable{}".format(i)] = variable[i]
            f_out = ne.evaluate(self.formula)
            f_out = f_out.astype("float32")
            np.nan_to_num(f_out, nan=MAXFLOAT, copy=False)
            return (f_out,)
if __name__ == "__main__":
    LOG.info(f"Starting job SeismicCalculation (pid {os.getpid()})")

    job = SeismicCalculation()
    job.report()
    job.run()
    