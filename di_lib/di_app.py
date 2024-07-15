from collections import namedtuple
from token import OP
from typing import Any, List, Optional, Tuple
import argparse
import requests
import json
import sys

from .seismic_cube import DISeismicCube
from .seismic_line import DISeismicLine
from .session import DISession
import logging
import abc
import numpy as np
from multiprocessing import Pool, cpu_count
import signal
import functools
import time

LOG = logging.getLogger(__name__)

class JobDescription(namedtuple("JobDescription", ["job_id", "project_id", "token", "server_url"])):
    __slots__ = ()

Frag = namedtuple('Frag', "no_i span_i, no_x span_x")

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
    LOG.info(f"Signal caught in child {signum}")
    sys.exit(signum)

def main_sigint_handler_with_pool(pool, signum, frame):
    LOG.info(f"Caught signal {signum} in main process")
    if pool:
        pool.terminate()
        LOG.info(f"Pool terminated")
    sys.exit(100)

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
        return self._margin

    @property
    def n_processes(self):
        if self._n_processes == 0:
            default_n_proceses = min(cpu_count(), 8)
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
    def  compute(self, f_in: Tuple[np.ndarray]) -> Tuple:
        pass

    def process_fragment(self, i, c_in, c_out, frag):
        def output_frag_if_not_none(w_out, f_out, f_coords):
            if w_out:
                w_out.write_fragment(f_coords[0], f_coords[2], f_out)

        tmp_f = c_in.get_fragment(*frag)
        if tmp_f is None:
            LOG.info(f"Skipped: {i} {frag}")
            return i, "SKIP"
        f_out = self.compute((tmp_f,))
        if DiApp.wrong_output_formats((tmp_f,), f_out):
            raise RuntimeError(f"Wrong output array format: shape or dtype do not coinside with input")
        for w,f in zip(c_out, f_out):
            output_frag_if_not_none(w, f, frag)
        LOG.info(f"Processed {i} {frag}")
        return i, "OK"

    def log_progress(self, stage: str, progress: int):
        url = f"{self.url}/jobs/progress/{self.job_id}/"
        body = json.dumps({"stage": stage, "progress": progress}).encode("utf8")
        with requests.post(
                    url, data=body, headers={"Content-Type": "application/json", "x-di-authorization": self.token}
                ) as resp:
            if resp.status_code != 200:
                LOG.error("Failed to set progress for job, response code %s", resp.status_code)
                raise RuntimeError(f"Failed to set progress for job, response code {resp.status_code}")

    def completion_callback(self, iterable):
        self.completed_frags += 1
        LOG.info(f"Completion: {self.completed_frags*100 // self.total_frags}")
        self.log_progress("calculation", self.completed_frags*100 // self.total_frags)

    def generate_process_arguments(self) -> List[Any]:
        return []

    @abc.abstractmethod
    def run(self):
        self.report()
        LOG.debug(f"job {self.job_id} starting")

        signal.signal(signal.SIGTERM, children_sigterm_handler)
        signal.signal(signal.SIGINT, children_sigterm_handler)

        t_start = time.perf_counter()
        job_params = self.description
        

        pool = Pool(self.n_processes)
        LOG.info(f"Processing pool created, size: {self.n_processes}")

        run_args = self.generate_process_arguments()

        handlr = functools.partial(main_sigint_handler_with_pool, pool)
        signal.signal(signal.SIGTERM, handlr)
        signal.signal(signal.SIGINT, handlr)

        res = []
        for i in run_args:
            res.append(pool.apply_async(self.process_fragment, i, callback=self.completion_callback))

        res_final = []
        for r in res:
            try:
                res_final.append(r.get())
            except Exception as ex:
                res_final.append(f"Exception {ex}")
        pool.close()
        pool.join()
        t_end = time.perf_counter()
        LOG.info(f"Finished in {t_end-t_start} sec")
        return res_final

    def report(self):
        """Reports current job's parameters"""
        LOG.info(f"Job parameters: {self.description}")

class DiAppSeismic3D(DiApp):

    def __init__(self, in_name_par: str, out_name_par: str, out_names: List[str]) -> None:
        super().__init__()
        self.in_name_par = in_name_par
        self.out_names = out_names
        self.out_name_par = out_name_par
        self.cube_in: Optional[DISeismicCube] = None

    def process_fragment(self, i, c_in, c_out, frag):
        def output_frag_if_not_none(w_out, f_out, f_coords):
            if w_out:
                w_out.write_fragment(f_coords[0], f_coords[2], f_out)

        tmp_f: np.ndarray = c_in.get_fragment(*frag)
        if tmp_f is None:
            LOG.info(f"Skipped: {i} {frag}")
            return i, "SKIP"
        f_out = self.compute((tmp_f,))
        if DiApp.wrong_output_formats((tmp_f,), f_out):
            raise RuntimeError(f"Wrong output array format: shape or dtype do not coinside with input")
        for w,f in zip(c_out, f_out):
            output_frag_if_not_none(w, f, frag)
        LOG.info(f"Processed {i} {frag}")
        return i, "OK"

    def open_input_dataset(self):
        geometry_name, name, name2 = self.description[self.in_name_par][0].split("\n")
        c = self.session.get_cube(geometry_name, name, name2)
        LOG.debug(f"{c=}")
        return c

    def create_output_datasets(self, cube_in: DISeismicCube):
        res = []
        for result_name in self.out_names:
            name = self.description[self.out_name_par]
            name2 = self.__class__.__name__ + f" ({result_name})"
            c_out = self.session.create_cube_writer_as_other(cube_in, name, name2)
            res.append(c_out)
        return res

    def generate_optimal_grid(self, c: DISeismicCube):
        incr_i = 64
        incr_x = 64
        grid = c.generate_fragments_grid_incr(incr_i, incr_x)
        return grid

    def run(self):
        self.report()
        LOG.debug(f"job {self.job_id} starting")

        signal.signal(signal.SIGTERM, children_sigterm_handler)
        signal.signal(signal.SIGINT, children_sigterm_handler)

        t_start = time.perf_counter()
        job_params = self.description
        
        c = self.open_input_dataset()
        self.cube_in = c
        
        w_out = self.create_output_datasets(c)

        pool = Pool(self.n_processes)
        LOG.info(f"Processing pool created, size: {self.n_processes}")

        n_frags = 10
        grid = self.generate_optimal_grid(c)
        LOG.debug(f"{grid=}")

        enum_grid = enumerate(grid)
        self.total_frags = len(grid)
        run_args = [(i[0], c, w_out, i[1]) for i in enum_grid]

        handlr = functools.partial(main_sigint_handler_with_pool, pool)
        signal.signal(signal.SIGTERM, handlr)
        signal.signal(signal.SIGINT, handlr)

        res = []
        for i in run_args:
            res.append(pool.apply_async(self.process_fragment, i, callback=self.completion_callback))

        res_final = []
        for r in res:
            try:
                res_final.append(r.get())
            except Exception as ex:
                res_final.append(f"Exception {ex}")
        pool.close()
        pool.join()
        t_end = time.perf_counter()
        LOG.info(f"Finished in {t_end-t_start} sec")
        return res_final

class DiAppSeismic3DMultiple(DiApp):
    """This class launches calculations on multiple input data belonging to the same geometry.
    """
    def __init__(self, in_name_par: str, out_name_par: str, out_names: List[str]) -> None:
        super().__init__()
        self.in_name_par = in_name_par
        self.out_names = out_names
        self.out_name_par = out_name_par
        self.cube_in: Optional[DISeismicCube] = None
        self.output_cubes_parameters = {}

    def process_fragment(self, i, c_in, c_out, frag):
        def output_frag_if_not_none(w_out, f_out, f_coords):
            if w_out:
                w_out.write_fragment(f_coords[0], f_coords[2], f_out)

        tmp_f = tuple([c.get_fragment(*frag) for c in c_in])
        for f in tmp_f:
            if f is None:
                LOG.info(f"Skipped: {i} {frag}")
                return i, "SKIP"
        f_out = self.compute(tmp_f)
        if DiApp.wrong_output_formats(tmp_f, f_out):
            raise RuntimeError(f"Wrong output array format: shape or dtype do not coinside with input")
        for w,f in zip(c_out, f_out):
            output_frag_if_not_none(w, f, frag)
        LOG.info(f"Processed {i} {frag}")
        return i, "OK"

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
                raise RuntimeError("Time axis parameters of the input cubbes do not coinside")
            if c.n_samples != c0.n_samples:
                raise RuntimeError("Time axis parameters of the input cubbes do not coinside")
            if c.data_start != c0.data_start:
                raise RuntimeError("Time axis parameters of the input cubbes do not coinside")
            if c.domain != c0.domain:
                raise RuntimeError("Time axis parameters of the input cubbes do not coinside")
        LOG.debug(f"{cubes=}")
        return cubes

    def create_output_datasets(self):
        cube_in = self.cubes_in[0]
        res = []
        for result_name in self.out_names:
            name = self.description[self.out_name_par]
            name2 = self.__class__.__name__ + f" ({result_name})"
            c_out = self.session.create_cube_writer_as_other(cube_in, name, name2, **self.output_cubes_parameters)
            res.append(c_out)
        self.output_cubes_parameters.update(res[0]._get_info())
        return res

    def generate_optimal_grid(self):
        """Currently the first cube in the input dataset is used to generate fragmens grig.
        """
        incr_i = 64
        incr_x = 64
        c = self.cubes_in[0]
        grid = c.generate_fragments_grid_incr(incr_i, incr_x)
        return grid

    def run(self):
        self.report()
        LOG.debug(f"job {self.job_id} starting")

        signal.signal(signal.SIGTERM, children_sigterm_handler)
        signal.signal(signal.SIGINT, children_sigterm_handler)

        t_start = time.perf_counter()
        job_params = self.description
        
        cubes_in = self.open_input_datasets()
        self.cubes_in = cubes_in
        
        w_out = self.create_output_datasets()

        pool = Pool(self.n_processes)
        LOG.info(f"Processing pool created, size: {self.n_processes}")

        n_frags = 10
        grid = self.generate_optimal_grid()
        LOG.debug(f"{grid=}")

        enum_grid = enumerate(grid)
        self.total_frags = len(grid)
        run_args = [(i[0], cubes_in, w_out, i[1]) for i in enum_grid]

        handlr = functools.partial(main_sigint_handler_with_pool, pool)
        signal.signal(signal.SIGTERM, handlr)
        signal.signal(signal.SIGINT, handlr)

        res = []
        for i in run_args:
            res.append(pool.apply_async(self.process_fragment, i, callback=self.completion_callback))

        res_final = []
        for r in res:
            try:
                res_final.append(r.get())
            except Exception as ex:
                res_final.append(f"Exception {ex}")
        pool.close()
        pool.join()
        t_end = time.perf_counter()
        LOG.info(f"Finished in {t_end-t_start} sec")
        return res_final


class DiAppSeismic3D2D(DiApp):
    """Process 3D cube and multiple seismic lines"""

    def __init__(self, in_name_par: str, in_line_names_par: str, out_name_par: str, out_names: List[str]) -> None:
        super().__init__()
        self.in_name_par = in_name_par
        self.in_line_names_par = in_line_names_par
        self.out_names = out_names
        self.out_name_par = out_name_par
        self.cube_in: Optional[DISeismicCube] = None
        self.lines_in = []

    def process_fragment(self, i, c_in, c_out, frag):
        def output_frag_if_not_none(w_out, f_out, f_coords):
            if w_out:
                w_out.write_fragment(f_coords[0], f_coords[2], f_out)


        frag_i = Frag(*frag)
        frag_e = enlarge_fragment(Frag(*frag), self.margin)

        tmp_f: Optional[np.ndarray] = c_in.get_fragment(*frag_e)
        if tmp_f is None:
            LOG.info(f"Skipped: {i} {frag_e=} {frag=}")
            return i, "SKIP"
        f_out = self.compute((tmp_f,))
        if DiApp.wrong_output_formats((tmp_f,), f_out):
            raise RuntimeError(f"Wrong output array format: shape or dtype do not coinside with input")
        for w,f in zip(c_out, f_out):
            ar_out = f[frag_i.no_i-frag_e.no_i:frag_e.span_i-self.margin, frag_i.no_x-frag_e.no_x:frag_e.span_x-self.margin,:]
            output_frag_if_not_none(w, ar_out, frag)
        LOG.info(f"Processed {i} {frag_e=} {frag=}")
        return i, "OK"

    def process_line_data(self, nm, p_in, p_out):
        def output_frag_if_not_none(w_out, f_out):
            if w_out:
                w_out.write_data(f_out)

        tmp_f: Optional[np.ndarray] = p_in.get_data()
        if tmp_f is None:
            LOG.info(f"Skipped: {nm}")
            return nm, "SKIP"
        f_out = self.compute((tmp_f,))
        if DiApp.wrong_output_formats((tmp_f,), f_out):
            raise RuntimeError(f"Wrong output array format: shape or dtype do not coinside with input")
        for w,f in zip(p_out, f_out):
            output_frag_if_not_none(w, f)
        LOG.info(f"Processed {nm}")
        return nm, "OK"


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
            name2 = self.__class__.__name__ + f" ({result_name})"
            c_out = self.session.create_cube_writer_as_other(cube_in, name, name2)
            res.append(c_out)
        return res

    def open_input_lines(self):
        names = [i.split("\n") for i in self.description[self.in_line_names_par]]
        return [self.session.get_line(n[0], n[1]) for n in names]

    def create_output_lines(self, lines_in: List[DISeismicLine]):
        res = []
        for p_in in lines_in:
            res1 = []
            for result_name in self.out_names:
                name = self.description[self.out_name_par] + f" ({p_in.name})"
                name2 = self.__class__.__name__ + f" ({result_name})"
                p_out = self.session.create_line_writer_as_other(p_in, name, name2)
                res1.append(p_out)
            res.append(res1)
        return res

    def generate_optimal_grid(self, c: DISeismicCube):
        incr_i = 64
        incr_x = 64
        grid = c.generate_fragments_grid_incr(incr_i, incr_x)
        return grid

    def run(self):
        self.report()
        LOG.debug(f"job {self.job_id} starting")

        signal.signal(signal.SIGTERM, children_sigterm_handler)
        signal.signal(signal.SIGINT, children_sigterm_handler)

        t_start = time.perf_counter()
        job_params = self.description
        
        run_args_cubes = []
        c = self.open_input_dataset()
        if c:
            self.cube_in = c
            
            w_out = self.create_output_datasets(c)


            grid = self.generate_optimal_grid(c)
            LOG.debug(f"{grid=}")

            enum_grid = enumerate(grid)
            self.total_frags = len(grid)
            run_args_cubes = [(i[0], c, w_out, i[1]) for i in enum_grid]

        self.lines_in = self.open_input_lines()
        self.total_frags += len(self.lines_in)
        output_lines = self.create_output_lines(self.lines_in)
        run_args_lines = [(str(i[0]), i[0], i[1]) for i in zip(self.lines_in, output_lines)]

        pool = Pool(self.n_processes)
        LOG.info(f"Processing pool created, size: {self.n_processes}")

        handlr = functools.partial(main_sigint_handler_with_pool, pool)
        signal.signal(signal.SIGTERM, handlr)
        signal.signal(signal.SIGINT, handlr)

        res = []
        for i in run_args_cubes:
            res.append(pool.apply_async(self.process_fragment, i, callback=self.completion_callback))

        for i in run_args_lines:
            res.append(pool.apply_async(self.process_line_data, i, callback=self.completion_callback))

        res_final = []
        for r in res:
            try:
                res_final.append(r.get())
            except Exception as ex:
                res_final.append(f"Exception {ex}")
        pool.close()
        pool.join()
        t_end = time.perf_counter()
        LOG.info(f"Finished in {t_end-t_start} sec")
        return res_final
