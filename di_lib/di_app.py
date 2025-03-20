from collections import namedtuple, Counter
from token import OP
from typing import Any, List, Optional, Tuple, Dict, Union, cast
import argparse
import requests
import json
import sys
import os

from .seismic_cube import DISeismicCube
from .seismic_line import DISeismicLine
from .session import DISession
import logging
import abc
import numpy as np
import multiprocessing
import signal
import functools
import time
import traceback
from importlib.metadata import distributions

from concurrent.futures import ProcessPoolExecutor, wait, Future, as_completed


LOG = logging.getLogger(__name__)
MAX_TRIES = 10

class JobDescription(namedtuple("JobDescription", ["job_id", "project_id", "token", "server_url"])):
    __slots__ = ()

class Frag(namedtuple("Frag", "no_i span_i no_x span_x")):
    __slots__ = ()

class Context(namedtuple("Context", "in_cube_params in_line_params out_cube_params out_line_params")):
    __slots__ = ()

class ProcessCParams(namedtuple("ProcessCParams", "c_in c_out frag")):
    __slots__ = ()

class ProcessLParams(namedtuple("ProcessLParams", "nm, p_in, p_out")):
    __slots__ = ()

def parse_args() -> JobDescription:
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", "--job", required=True, help="job ID")
    parser.add_argument("-P", "--project", required=True, help="project id")
    parser.add_argument("-t", "--token", required=True, help="authorization token")
    parser.add_argument("-u", "--url", required=True, help="DB server base URL")
    parser.add_argument("-p", "--port", required=True, type=int, help="DB server port")
    args = parser.parse_args()
    LOG.debug(f"{args}")
    full_url = f"{args.url}:{args.port}"
    res = JobDescription(args.job, args.project, args.token, full_url)
    LOG.debug(f"{res}")
    return res


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


class DiApp(metaclass=abc.ABCMeta):

    def __init__(self) -> None:
        descr = parse_args()
        self.job_id = descr.job_id
        self.project_id = descr.project_id
        self.token = descr.token
        self.url = descr.server_url
        self._description = None
        self.session = DISession(self.project_id, self.url, self.token)
        self._n_processes = 0
        self.total_frags = 0
        self.completed_frags = 0
        self._margin = None
        self.out_data_params = {"job_id": self.job_id}
        self.loop_no = 0

    @property
    def description(self):
        if self._description is None:
            with requests.get(f"{self.url}/jobs/info/{self.job_id}/") as resp:
                if resp.status_code != 200:
                    LOG.error(f"unable to get job description from {self.url} for {self.job_id}")
                    raise RuntimeError(f"unable to get job description from {self.url} for {self.job_id}")
                resp_j = json.loads(resp.content)
                self._description = resp_j["job_description"]
        return self._description

    @property
    def margin(self):
        if self._margin is None:
            self._margin = self.description.get("margin", 0)
            LOG.debug(f"Margin set to {self._margin}")
        return self._margin

    @property
    def n_processes(self):
        if self._n_processes == 0:
            default_n_proceses = min(multiprocessing.cpu_count(), 8)
            self._n_processes = self.description.get("Threads Count", default_n_proceses)
        return self._n_processes

    @staticmethod        
    def wrong_output_formats(in_arrays: Tuple[np.ndarray], out_arrays: Tuple[np.ndarray]):
        in_array = in_arrays[0] # !!! TBD: we should check if all input formats are same
        if not all(i.dtype == in_array.dtype for i in out_arrays):
            LOG.error(f"Wrong dtype {in_array.dtype} {[a.dtype for a in out_arrays]}")
            return True
        # Shapes of output arrays must coinside with shapes of input arrays excluding last (time/deph) axis
        if not all(i.shape[:-1] == in_array.shape[:-1] for i in out_arrays):
            LOG.error(f"Wrong shape {in_array.shape} {[a.shape for a in out_arrays]}")
            return True
        # TBD!!!: should check if number of samples (last number in shape) is correct!

    @abc.abstractmethod
    def  compute(self, f_in: Tuple[np.ndarray], context: Context=Context(None, None, None, None)) -> Tuple:
        pass

    def process_cube_data(self, task_id: int, params: ProcessCParams) -> Tuple[int, str]:
        raise NotImplementedError("processing cube data not implemented")
    
    def process_line_data(self, task_id: int, params: ProcessLParams) -> Tuple[int, str]:
        raise NotImplementedError("processing line data not implemented")

    def process_fragment(self, task_id, params: Union[ProcessCParams, ProcessLParams]) -> Tuple[int, str]:
        try:
            if issubclass(type(params), ProcessCParams):
                return self.process_cube_data(task_id, cast(ProcessCParams, params))
            elif issubclass(type(params), ProcessLParams):
                return self.process_line_data(task_id, cast(ProcessLParams, params))
            else:
                raise RuntimeError(f"Unexpected type of argument {type(params)}")
        except Exception as ex:
            traceback.print_exc()
            ex.args = (task_id,) + ex.args
            raise ex

    def log_progress(self, stage: str, progress: int):
        url = f"{self.url}/jobs/progress/{self.job_id}/"
        body = json.dumps({"stage": stage, "progress": progress}).encode("utf8")
        with requests.post(
                    url, data=body, headers={"Content-Type": "application/json", "x-di-authorization": self.token}
                ) as resp:
            if resp.status_code != 200:
                LOG.error("Failed to set progress for job, response code %s", resp.status_code)
                raise RuntimeError(f"Failed to set progress for job {self.job_id}, response code {resp.status_code}")

    def completion_callback(self, iterable):
        self.completed_frags += 1
        LOG.debug(f"Completion: {self.completed_frags*100 // self.total_frags}")
        completed = False
        n_iter = 0
        while not completed and n_iter < 10: # TODO: should eliminate magic constant!
            try:
                self.log_progress("calculation", self.completed_frags*100 // self.total_frags)
                completed = True
            except RuntimeError as ex:
                LOG.error(f"Failed to set progress for job, cause {ex}")
                raise ex
            except Exception as ex:
                LOG.error(f"Failed to set progress for job, cause {ex}")
            n_iter += 1

    @abc.abstractmethod
    def generate_process_arguments(self) -> Dict[int, Union[ProcessCParams, ProcessLParams]]:
        return {}

    def run(self):
        self.report()
        LOG.debug(f"job {self.job_id} starting")

        t_start = time.perf_counter()
        
        run_args = self.generate_process_arguments()

        with ProcessPoolExecutor(max_workers=self.n_processes, initializer=init_process) as executor:
            LOG.info(f"Executor created, size: {self.n_processes}")

            handlr = functools.partial(main_sigint_handler_with_exec, executor)
            signal.signal(signal.SIGTERM, handlr)
            signal.signal(signal.SIGINT, handlr)
            loop_number = 0
            res_final = []
            while (len(run_args) != 0) and (loop_number < MAX_TRIES):
                loop_number += 1
                futures: List[Future] = []
                for i in run_args.items():
                    f = executor.submit(self.process_fragment, *i)
                    futures.append(f)

                LOG.debug(f"Waiting for completion of {len(futures)} futures in {loop_number=}")
                for future in as_completed(futures):
                    try:
                        res = future.result()
                        res_final.append(res)
                        del run_args[res[0]]
                        self.completion_callback(None)
                    except RuntimeError as ex:
                        LOG.error(f"Task failed, no restart: {type(ex)} {ex}")
                        LOG.error(f"Exception {type(ex)}: Failed job id: {ex.args[0]} params: {run_args[ex.args[0]]}")
                        del run_args[ex.args[0]]
                        res_final.append((ex.args[0], f"Exception final: {type(ex)} {ex}"))
                        self.completion_callback(None)
                    except Exception as ex:
                        LOG.debug(f"Exception for resubmit: {type(ex)} {ex}")
                        LOG.debug(f"Exception {type(ex)}: Failed job id: {ex.args[0]} params: {run_args[ex.args[0]]}")

        t_end = time.perf_counter()
        if run_args:
            LOG.debug(f"Still not computed after {MAX_TRIES} tries: {run_args}")
            LOG.error(f"Still not computed after {MAX_TRIES} tries: {len(run_args)}")
        LOG.info(f"Finished in {t_end-t_start} sec")
        report = self.make_result_report(res_final)
        return report

    def report(self):
        """Reports current job's parameters"""
        LOG.info(f"Job parameters: {self.description}")
        LOG.info(f"Platform: {sys.platform} Version: {sys.version}")
        installed_packages = {dist.metadata["Name"]: dist.version for dist in distributions()}
        freeze = [f"{p}=={v}" for p,v in installed_packages.items()]
        LOG.info(f"Installed packages: {freeze}")
    
    def make_result_report(self, result: List[Tuple[int, Any]]) -> str:
        """Reports result of the job"""
        LOG.debug(f"Job result: {result}")
        summary = Counter([r[1] for r in result])
        LOG.debug(f"Job result: {summary}")
        return f"{summary}"
    
class DiAppSeismic3D(DiApp):

    def __init__(self, in_name_par: str, out_name_par: str, out_names: List[str]) -> None:
        super().__init__()
        self.in_name_par = in_name_par
        self.out_names = out_names
        self.out_name_par = out_name_par
        self.cube_in: Optional[DISeismicCube] = None
        # self.out_data_params = {}

    def open_input_dataset(self):
        geometry_name, name, name2 = self.description[self.in_name_par][0].split("\n")
        c = self.session.get_cube(geometry_name, name, name2)
        LOG.debug(f"{c=}")
        return c

    def create_output_datasets(self, cube_in: DISeismicCube):
        res = []
        for result_name in self.out_names:
            name = self.description[self.out_name_par]
            # name2 = self.__class__.__name__ + f" ({result_name})"
            name2 = f"{result_name}"
            c_out = self.session.create_cube_writer_as_other(cube_in, name, name2, **self.out_data_params)
            res.append(c_out)
        return res

    def generate_optimal_grid(self, c: DISeismicCube):
        incr_i = 64
        incr_x = 64
        grid = c.generate_fragments_grid_incr(incr_i, incr_x)
        return grid

    def process_cube_data(self, task_id: int, params: ProcessCParams) -> Tuple[int, str]:
        def output_frag_if_not_none(w_out, f_out, f_coords):
            if w_out:
                w_out.write_fragment(f_coords[0], f_coords[2], f_out)

        tmp_f: np.ndarray = params.c_in.get_fragment(*params.frag)
        if tmp_f is None:
            LOG.debug(f"Skipped: {task_id} {params.frag}")
            return task_id, "SKIP"
        out_cube_params = params.c_out[0]._get_info() if len(params.c_out) else None
        context = Context(in_cube_params=params.c_in._get_info(), in_line_params=None, out_cube_params=out_cube_params, out_line_params=None)
        context.in_cube_params["chunk"] = params.frag
        f_out = self.compute((tmp_f,), context=context)
        if DiApp.wrong_output_formats((tmp_f,), f_out):
            raise RuntimeError(f"Wrong output array format: shape or dtype do not coincide with input")
        for w,f in zip(params.c_out, f_out):
            output_frag_if_not_none(w, f, params.frag)
        LOG.debug(f"Processed {task_id} {params.frag}")
        return task_id, "OK"

    def generate_process_arguments(self) -> Dict[int, ProcessCParams]:
        c = self.open_input_dataset()
        self.cube_in = c
        
        w_out = self.create_output_datasets(c)

        grid = self.generate_optimal_grid(c)
        LOG.debug(f"{grid=}")

        enum_grid = enumerate(grid)
        self.total_frags = len(grid)
        run_args = {i[0]: ProcessCParams(c, w_out, i[1]) for i in enum_grid}
        return run_args

class DiAppSeismic3DMultiple(DiApp):
    """This class launches calculations on multiple input data belonging to the same geometry.
    """
    def __init__(self, in_name_par: str, out_name_par: str, out_names: List[str]) -> None:
        super().__init__()
        self.in_name_par = in_name_par
        self.out_names = out_names
        self.out_name_par = out_name_par
        self.cube_in: Optional[DISeismicCube] = None

    def process_cube_data(self, task_id, params: ProcessCParams) -> Tuple[int, str]:
        def output_frag_if_not_none(w_out, f_out, f_coords):
            if w_out:
                w_out.write_fragment(f_coords[0], f_coords[2], f_out)

        tmp_f = tuple([c.get_fragment(*params.frag) for c in params.c_in])
        for f in tmp_f:
            if f is None:
                LOG.debug(f"Skipped: {task_id} {params.frag}")
                return task_id, "SKIP"
        out_cube_params = params.c_out[0]._get_info() if len(params.c_out) else None
        context = Context(in_cube_params=params.c_in[0]._get_info(), in_line_params=None, out_cube_params=out_cube_params, out_line_params=None)
        context.in_cube_params["chunk"] = params.frag
        f_out = ()
        f_out = self.compute(tmp_f, context=context)
        if DiApp.wrong_output_formats(tmp_f, f_out):
            raise RuntimeError(f"Wrong output array format: shape or dtype do not coinside with input")
        for w,f in zip(params.c_out, f_out):
            output_frag_if_not_none(w, f, params.frag)
        LOG.debug(f"Processed {task_id} {params.frag}")
        return task_id, "OK"

    def open_input_datasets(self):
        cubes_names = [i.split("\n") for i in self.description[self.in_name_par]]
        # Check if all cubes belong to the same geometry
        if len(set(i[0] for i in cubes_names)) != 1:
            raise RuntimeError(f"All cubes must belong to the same geometry!")
        cubes = [self.session.get_cube(cn[0], cn[1], cn[2]) for cn in cubes_names]
        # Check that all input cubes have the same time-axis parameters
        c0 = cubes[0]
        for c in cubes:
            if c.time_step != c0.time_step:
                raise RuntimeError("Time axis parameters of the input cubes do not coincide")
            if c.n_samples != c0.n_samples:
                raise RuntimeError("Time axis parameters of the input cubes do not coincide")
            if c.data_start != c0.data_start:
                raise RuntimeError("Time axis parameters of the input cubes do not coincide")
            if c.domain != c0.domain:
                raise RuntimeError("Time axis parameters of the input cubes do not coincide")
        LOG.debug(f"{cubes=}")
        return cubes

    def create_output_datasets(self):
        cube_in = self.cubes_in[0]
        res = []
        for result_name in self.out_names:
            name = self.description[self.out_name_par]
            # name2 = self.__class__.__name__ + f" ({result_name})"
            name2 = f"{result_name}"
            c_out = self.session.create_cube_writer_as_other(cube_in, name, name2, **self.out_data_params)
            res.append(c_out)
        return res

    def generate_optimal_grid(self):
        """Currently the first cube in the input dataset is used to generate fragmens grig.
        """
        incr_i = 64
        incr_x = 64
        c = self.cubes_in[0]
        grid = c.generate_fragments_grid_incr(incr_i, incr_x)
        return grid

    def generate_process_arguments(self) -> Dict[int, ProcessCParams]:
        cubes_in = self.open_input_datasets()
        self.cubes_in = cubes_in
        
        w_out = self.create_output_datasets()
        grid = self.generate_optimal_grid()
        LOG.debug(f"{grid=}")

        enum_grid = enumerate(grid)
        self.total_frags = len(grid)
        # run_args = [(i[0], cubes_in, w_out, i[1]) for i in enum_grid]
        run_args = {i[0]: ProcessCParams(cubes_in, w_out, i[1]) for i in enum_grid}
        return run_args

class DiAppSeismic3D2D(DiApp):
    """Process 3D cube and multiple seismic lines"""

    def __init__(self, in_name_par: str, in_line_geometries_par: str, in_line_names_par: str, out_name_par: str, out_names: List[str]) -> None:
        super().__init__()
        self.in_name_par = in_name_par
        self.in_line_geometries_par = in_line_geometries_par
        self.in_line_names_par = in_line_names_par
        self.out_names = out_names
        self.out_name_par = out_name_par
        self.cube_in: Optional[DISeismicCube] = None
        self.lines_in = []
        # self.out_data_params = {}

    def process_cube_data(self, task_id: int, params: ProcessCParams) -> Tuple[int, str]:
        def output_frag_if_not_none(w_out, f_out, f_coords):
            if w_out:
                w_out.write_fragment(f_coords[0], f_coords[2], f_out)


        frag_i = Frag(*params.frag)
        frag_e = enlarge_fragment(Frag(*params.frag), self.margin)
        LOG.debug(f"Start processing {task_id} {frag_e=} {params.frag=}")
        tmp_f: Optional[np.ndarray] = params.c_in.get_fragment(*frag_e)
        if tmp_f is None:
            LOG.debug(f"Skipped: {task_id} {frag_e=} {params.frag=}")
            return task_id, "SKIP"
        out_cube_params = params.c_out[0]._get_info() if len(params.c_out) else None
        context = Context(in_cube_params=params.c_in._get_info(), in_line_params=None, out_cube_params=out_cube_params, out_line_params=None)
        context.in_cube_params["chunk"] = params.frag
        f_out = self.compute((tmp_f,), context=context)
        if DiApp.wrong_output_formats((tmp_f,), f_out):
            raise RuntimeError(f"Wrong output array format: shape or dtype do not coincide with input")
        for w,f in zip(params.c_out, f_out):
            ar_out = f[frag_i.no_i-frag_e.no_i:frag_e.span_i-self.margin, frag_i.no_x-frag_e.no_x:frag_e.span_x-self.margin,:]
            output_frag_if_not_none(w, ar_out, params.frag)
        LOG.debug(f"Processed {task_id} {frag_e=} {params.frag=}")
        return task_id, "OK"

    def process_line_data(self, task_id: int, params: ProcessLParams) -> Tuple[int, str]:
        def output_frag_if_not_none(w_out, f_out):
            if w_out:
                w_out.write_data(f_out)

        tmp_f: Optional[np.ndarray] = params.p_in.get_data()
        if tmp_f is None:
            LOG.debug(f"Skipped: {params.nm}")
            return task_id, "SKIP"
        out_line_params = params.p_out[0]._get_info() if len(params.p_out) else None
        context = Context(in_cube_params=None, in_line_params=params.p_in._get_info(), out_cube_params=None, out_line_params=out_line_params)
        f_out = self.compute((tmp_f,), context=context)
        if DiApp.wrong_output_formats((tmp_f,), f_out):
            raise RuntimeError(f"Wrong output array format: shape or dtype do not coincide with input")
        for w,f in zip(params.p_out, f_out):
            output_frag_if_not_none(w, f)
        LOG.debug(f"Processed {params.nm}")
        return task_id, "OK"


    def open_input_dataset(self):
        if len(self.description[self.in_name_par]) == 0:
            return
        geometry_name, name, name2 = self.description[self.in_name_par][0].split("\n")
        c = self.session.get_cube(geometry_name, name, name2)
        LOG.debug(f"{c=}")
        return c

    def create_output_datasets(self, cube_in: DISeismicCube):
        res = []
        for result_name in self.out_names:
            name = self.description[self.out_name_par]
            # name2 = self.__class__.__name__ + f" ({result_name})"
            name2 = f"{result_name}"
            c_out = self.session.create_cube_writer_as_other(cube_in, name, name2, **self.out_data_params)
            res.append(c_out)
        return res

    def open_input_lines(self):
        # names = [i.split("\n") for i in self.description[self.in_line_names_par]]
        geometries = self.description[self.in_line_geometries_par]
        line_data_names = self.description[self.in_line_names_par]
        names = [(g,*n.split("\n")) for g in geometries for n in line_data_names]
        # Input profiles names are [geom, name, name2]
        return [self.session.get_line(n[0], n[1], n[2]) for n in names]

    def create_output_lines(self, lines_in: List[DISeismicLine]):
        res = []
        for p_in in lines_in:
            res1 = []
            for result_name in self.out_names:
                name = self.description[self.out_name_par] + f" ({p_in.name})"
                # name2 = self.__class__.__name__ + f" ({result_name})"
                name2 = f"{result_name}"
                p_out = self.session.create_line_writer_as_other(p_in, name, name2, **self.out_data_params)
                res1.append(p_out)
            res.append(res1)
        return res

    def generate_optimal_grid(self, c: DISeismicCube):
        incr_i = 64
        incr_x = 64
        grid = c.generate_fragments_grid_incr(incr_i, incr_x)
        return grid


    def generate_process_arguments(self) -> Dict[int, Union[ProcessCParams, ProcessLParams]]:
        run_args: Dict[int, Union[ProcessCParams, ProcessLParams]] = {}
        c = self.open_input_dataset()

        if c:
            self.cube_in = c
            
            w_out = self.create_output_datasets(c)


            grid = self.generate_optimal_grid(c)
            LOG.debug(f"{grid=}")

            enum_grid = enumerate(grid)
            run_args = {i[0]: ProcessCParams(c, w_out, i[1]) for i in enum_grid}

        self.lines_in = self.open_input_lines()
        output_lines = self.create_output_lines(self.lines_in)
        run_args_lines = {i[0]: ProcessLParams(str(i[1][0]), i[1][0], i[1][1]) for i in enumerate(zip(self.lines_in, output_lines), len(run_args))}
        run_args.update(run_args_lines)
        self.total_frags = len(run_args)
        return run_args
