import math
import logging
from typing import Optional, Tuple, List
import numpy as np
import requests
import json
import struct
from dataclasses import dataclass

LOG = logging.getLogger(__name__)

SAMPLE_BYTE_LEN = 4 # Corresponds to lsb format of data
MAXFLOAT = 3.40282347e+38 ## stands for undefined values of parameters
MAXFLOAT09 = 0.9*3.40282347e+38 ## stands for undefined values of parameters

def scalar_prod(xx, yy):
    return sum(x*y for x, y in zip(xx, yy))

def subtract(x, y):
    return tuple((xx-yy for xx, yy in zip(x, y)))

def add(x, y):
    return tuple((xx+yy for xx, yy in zip(x, y)))

def mult_by_scalar(x, a):
    return tuple((xx*a for xx in x))

def round(x):
    return math.floor(x+0.5)

#######  Utility functions ####

def join_time_axes(t1, t2):
    "Builds the joint time axis from two axes t1 and t2. t1, t2: (t_origin, t_step, n)"
    new_origin = min(t1[0], t2[0])
    new_step = min(t1[1], t2[1])
    # adjust origin, so that it would be multiple of new step
    new_origin = math.ceil(new_origin / new_step) * new_step
    new_n = int(math.floor((max((t1[0] + t1[1]*(t1[2]-1)), (t2[0] + t2[1]*(t2[2])-1)) - new_origin) / new_step)) + 1
    return (new_origin, new_step, new_n)

def recalculate_trace_to_new_time_axis(trace, t1, t_new):
    "Recalculates trace from one time grid to another doing interpolation if needed"
    def trace_value(t):
        ind = (t - t1[0]) / t1[1]
        if (ind < 0) or (ind > t1[2]-1):
            return MAXFLOAT
        alpha, i = math.modf(ind)
        i = int(i)
        if i == t1[2]-1:
            return trace[i]
        if (trace[i] > MAXFLOAT09) or (trace[i+1] > MAXFLOAT09):
            return MAXFLOAT
        return trace[i]*(1.-alpha) + trace[i+1]*alpha
    return tuple(trace_value(t_new[0] + i*t_new[1]) for i in range(t_new[2]))

def generate_fragments_grid_incr(min_i, n_i, inc_i, min_x, n_x, inc_x):
    """Generate grid with constant increments over inlines/xlines.
    """
    res = [(i, j, min(n_i-1, i+inc_i-1), min(n_x-1, j+inc_x-1)) for i in range(min_i, n_i-1, inc_i) for j in range(min_x, n_x-1, inc_x)]
    return res

def generate_fragments_grid(min_i, n_i, nfrags_i, min_x, n_x, nfrags_x):
    # inlines/xlines are counted starting from 1 by default
    if min_i is None:
        min_i = 1
    if min_x is None:
        min_x = 1
    inc_i = (n_i - min_i) // nfrags_i
    inc_x = (n_x - min_x) // nfrags_x
    return generate_fragments_grid_incr(min_i, n_i, inc_i, min_x, n_x, inc_x)

@dataclass(frozen=True)
class DIGeometryInfo:
    name: str
    id: int
    origin: Tuple[float, float]
    v_i: Tuple[float, float]
    v_x: Tuple[float, float]
    ts: str
    owner: str

class DIGeometry:
    def __init__(self, project_id: int, name: str) -> None:
        self.server_url = ""
        self.token = ""
        self.project_id = project_id
        self.name = name
        self._geometry_info = None

    @property
    def info(self) -> DIGeometryInfo:
        if self._geometry_info.id is None:
            self._read_info()
        return self._geometry_info

    def _read_info(self) -> None:
        with requests.get(f"{self.server_url}/seismic_3d/geometries/{self.project_id}/") as resp:
            if resp.status_code != 200:
                LOG.error("Caught exception during GET: %s", resp.content)
                raise RuntimeError(f"Caught exception during GET: {resp.content}")
            resp_j = json.loads(resp.content)
            LOG.debug("Reply: %s", resp_j)
            for geom in resp_j:
                if geom["name"] == self.name:
                    self._geometry_info = DIGeometryInfo(geom["name"], geom["id"], geom["origin"], geom["d_inline"], geom["d_xline"], ts=None, owner=None)
                    return
            raise ValueError(f"Geometry {self.name} not found")

class DISeismicCube:
    def __init__(self, project_id: int, geometry_name: str, name: str, name2: str) -> None:
        self.server_url = ""
        self.token = ""
        self.origin = None
        self.v_i = None
        self.v_x = None
        self.n_i = -1
        self.n_x = -1
        self.n_samples = -1
        self.time_step = 0
        self.norm_v_i = None
        self.norm_v_x = None
        self.data_start = None
        self.domain = "No domain"
        self.project_id = project_id
        self.geometry_name = geometry_name
        self.name = name
        self.name2 = name2
        self.geometry_id = None
        self.cube_id = None
        self.min_i = -1
        self.min_x = -1

    def __repr__(self):
        return f"DISeismicCube: {self.cube_id=} {self.geometry_name=} {self.name=} {self.name2=}"

    def _read_info(self) -> None:
        with requests.get(f"{self.server_url}/seismic_3d/list/{self.project_id}/") as resp:
            if resp.status_code != 200:
                LOG.error("Cant' get list of cubes: %s", resp.status_code)
                return None
            resp_j = json.loads(resp.content)
            for i in resp_j:
                if (i["geometry_name"] == self.geometry_name) and (i["name"] == self.name) and (i["name2"] == self.name2):
                    self.cube_id = i["id"]
                    self.geometry_id = i["geometry_id"]
                    LOG.debug(f"{i}")
                    self.origin = i["origin"]
                    self.v_i = i["d_inline"]
                    self.v_x = i["d_xline"]
                    self.n_i = i["max_inline"]+1
                    self.n_x = i["max_xline"]+1
                    self.n_samples = i["nz"]
                    self.time_step = i["z_step"]
                    self.domain = i["domain"]
                    self.data_start = i["z_start"]
                    self.min_i = i["min_inline"] or 1
                    self.min_x = i["min_xline"] or 1
            if self.cube_id is None:
                raise RuntimeError(f"Cube {self.name}/{self.name2} not found in {self.project_id=}")

    def _get_info(self):
        i = {
            "geometry_name": self.geometry_name,
            "geometry_id": self.geometry_id,
            "name": self.name,
            "name2": self.name2,
            "max_inline": self.n_i-1,
            "max_xline": self.n_x-1,
            "nz": self.n_samples,
            "origin": self.origin,
            "d_inline": self.v_i,
            "d_xline": self.v_x,
            "domain": self.domain,
            "z_start": self.data_start,
            "z_step": self.time_step,
            "min_inline": self.min_i,
            "min_xline": self.min_x,
            "id": self.cube_id
        }
        return i

    def get_fragment(self, inline_no: int, inline_count: int, xline_no: int, xline_count: int) -> Optional[np.ndarray]:
        url = f"{self.server_url}/seismic_3d/data/rect_fragment/{self.cube_id}/?inline_no={inline_no}&inline_count={inline_count}&xline_no={xline_no}&xline_count={xline_count}"
        with requests.get(url) as resp:
            bytes_read = len(resp.content)
            raw_data = resp.content
            if resp.status_code != 200:
                LOG.error("Request finished with error: %s", resp.status_code)
            nz, ncdps, ninlines = struct.unpack("<iii", raw_data[:12])
            LOG.debug(f"{nz}, {ncdps}, {ninlines}")
            if nz < 0:
                return None
            gr_arr = np.frombuffer(raw_data[12:], dtype=np.float32)
            gr_arr.shape = (ninlines, ncdps, nz)
            return gr_arr

    def get_fragment_z(self, inline_no: int, inline_count: int, xline_no: int, xline_count: int, z_no: int, z_count: int) -> Optional[np.ndarray]:
        url = f"{self.server_url}/seismic_3d/data/rect_fragment_z/{self.cube_id}/?inline_no={inline_no}&inline_count={inline_count}&xline_no={xline_no}&xline_count={xline_count}&z_no={z_no}&z_count={z_count}"
        with requests.get(url) as resp:
            bytes_read = len(resp.content)
            raw_data = resp.content
            if resp.status_code != 200:
                LOG.error("Request finished with error: %s", resp.status_code)
            nz, ncdps, ninlines = struct.unpack("<iii", raw_data[:12])
            LOG.debug(f"{nz}, {ncdps}, {ninlines}")
            if nz < 0:
                return None
            gr_arr = np.frombuffer(raw_data[12:], dtype=np.float32)
            gr_arr.shape = (ninlines, ncdps, nz)
            return gr_arr

    def get_inline(self, inline_no: int, trimmed: bool=True, top_ind: Optional[int]=None, bottom_ind: Optional[int] = None) -> np.ndarray:
        url = f"{self.server_url}/seismic_3d/data/inline/{self.cube_id}/?inline_no={inline_no}"
        if trimmed:
            url = f"{self.server_url}/seismic_3d/data/inline_trim/{self.cube_id}/?inline_no={inline_no}"
        if top_ind is not None:
            url += f"&top_no={top_ind}"
        if bottom_ind is not None:
            url += f"&bottom_no={bottom_ind}"
        with requests.get(url) as resp:
            bytes_read = len(resp.content)
            raw_data = resp.content
            if resp.status_code != 200:
                LOG.error("Request finished with error: %s", resp.status_code)
            start_i = 0
            if trimmed:
                start_i, nz, ninlines = struct.unpack("<iii", raw_data[:12])
            else:
                nz, ninlines = struct.unpack("<ii", raw_data[:8])
            LOG.debug(f"{nz=}, {ninlines=} {start_i=}")
            if nz < 0:
                return np.ndarray(0, dtype=np.float32)
            if trimmed:            
                gr_arr_p = np.frombuffer(raw_data[12:], dtype=np.float32)
                gr_arr_p.shape = (ninlines, nz)
                gr_arr = np.vstack([np.full((start_i, nz), np.nan, dtype=np.float32), gr_arr_p])
            else:
                gr_arr = np.frombuffer(raw_data[8:], dtype=np.float32)
                gr_arr.shape = (ninlines, nz)
            return gr_arr


    def get_xline(self, xline_no: int, trimmed: bool=True, top_ind: Optional[int]=None, bottom_ind: Optional[int] = None) -> np.ndarray:
        url = f"{self.server_url}/seismic_3d/data/xline/{self.cube_id}/?xline_no={xline_no}"
        if trimmed:
            url = f"{self.server_url}/seismic_3d/data/xline_trim/{self.cube_id}/?xline_no={xline_no}"
        if top_ind is not None:
            url += f"&top_no={top_ind}"
        if bottom_ind is not None:
            url += f"&bottom_no={bottom_ind}"
        with requests.get(url) as resp:
            bytes_read = len(resp.content)
            raw_data = resp.content
            if resp.status_code != 200:
                LOG.error("Request finished with error: %s", resp.status_code)
            start_i = 0
            if trimmed:
                start_i, nz, nxlines = struct.unpack("<iii", raw_data[:12])
            else:
                nz, nxlines = struct.unpack("<ii", raw_data[:8])
            LOG.debug(f"{nz=}, {nxlines=} {start_i=}")
            if nz < 0:
                return np.ndarray(0, dtype=np.float32)
            if trimmed:            
                gr_arr_p = np.frombuffer(raw_data[12:], dtype=np.float32)
                gr_arr_p.shape = (nxlines, nz)
                gr_arr = np.vstack([np.full((start_i, nz), np.nan, dtype=np.float32), gr_arr_p])
            else:
                gr_arr = np.frombuffer(raw_data[8:], dtype=np.float32)
                gr_arr.shape = (nxlines, nz)
            return gr_arr

    def get_timeline(self, z_no: int, trimmed: bool=True):
        url = f"{self.server_url}/seismic_3d/data/const_z/{self.cube_id}/?z_no={z_no}"
        if trimmed:
            url = f"{self.server_url}/seismic_3d/data/const_z_trim/{self.cube_id}/?z_no={z_no}"
        with requests.get(url) as resp:
            # bytes_read = len(resp.content)
            raw_data = resp.content
            if resp.status_code != 200:
                LOG.error(f"Request finished with error {resp.status_code}")
            start_i = 0
            if trimmed:
                start_i, start_x, nxlines, ninlines = struct.unpack("<iiii", raw_data[:16])
            else:
                nxlines, ninlines = struct.unpack("<ii", raw_data[:8])
            LOG.debug(f"{nxlines=}, {ninlines=} {start_i=} {start_x=}")
            if trimmed:            
                gr_arr_p = np.frombuffer(raw_data[16:], dtype=np.float32)
                gr_arr_p.shape = (ninlines, nxlines)
                # gr_arr = np.vstack([np.full((start_i, nxlines), np.nan, dtype=np.float32), gr_arr_p])
                gr_arr = gr_arr_p
            else:
                gr_arr = np.frombuffer(raw_data[8:], dtype=np.float32)
                gr_arr.shape = (ninlines, nxlines)
            return gr_arr

    def get_slice_1horizon(self, horizon_name: str, shift: float):
        url = f"{self.server_url}/seismic_3d/data/slice_1horizon/{self.cube_id}/"
        with requests.get(url, params={"horizon_top": horizon_name, "shift": shift}) as resp:
            # bytes_read = len(resp.content)
            raw_data = resp.content
            if resp.status_code != 200:
                LOG.error(f"Request finished with error {resp.status_code}")
            n_layers, min_nx, min_ny, nxlines, ninlines = struct.unpack("<iiiii", raw_data[:20])
            gr_arr = np.frombuffer(raw_data[20:], dtype=np.float32)
            gr_arr.shape = (n_layers, ninlines, nxlines)
            return gr_arr

    def generate_fragments_grid(self, nfrag_i, nfrag_x):
        tmp = generate_fragments_grid(self.min_i, self.n_i, nfrag_i, self.min_x, self.n_x, nfrag_x)
        return [(i[0], i[2]-i[0]+1, i[1], i[3]-i[1]+1) for i in tmp]

    def generate_fragments_grid_incr(self, incr_i, incr_x):
        tmp = generate_fragments_grid_incr(self.min_i, self.n_i, incr_i, self.min_x, self.n_x, incr_x)
        return [(i[0], i[2]-i[0]+1, i[1], i[3]-i[1]+1) for i in tmp]
    
    def get_statistics_for_horizons(self, hor_top_name, hor_bottom_name):
        url = f"{self.server_url}/seismic_3d/data/statistics/{self.cube_id}/"
        with requests.get(url, params={"horizon_top": hor_top_name, "horizon_bottom": hor_bottom_name}) as resp:
            if resp.status_code != 200:
                LOG.error("Cant' get list of cubes: %s", resp.status_code)
                return None
            resp_j = json.loads(resp.content)
            return resp_j
        
    def save_statistics_for_horizons(self, hor_top_name, hor_bottom_name, stat):
        url = f"{self.server_url}/seismic_3d/data/statistics/{self.cube_id}/"
        res_status = 200
        res_id = -1
        try:
            stat["hor_top"] = hor_top_name
            stat["hor_bottom"] = hor_bottom_name
            LOG.debug(f"{stat=}")
            body = json.dumps(stat).encode("utf8")
            with requests.post(
                    url, data=body, headers={"Content-Type": "application/json", "x-di-authorization": self.token}
                ) as resp:
                if resp.status_code != 200:
                    LOG.error("Failed to save statistics, response code %s", resp.status_code)
                    raise RuntimeError(f"Failed to save statistics, response code {resp.status_code}")
                resp_j = json.loads(resp.content)
                LOG.debug("Reply: %s", resp_j)
        except requests.exceptions.ConnectionError as ex:
            LOG.error("Exception during POST: %s", str(ex))
            raise ex

class DISeismicCubeWriter(DISeismicCube):
    def __init__(self, project_id: int, geometry_name: str, name: str, name2: str) -> None:
        super().__init__(project_id, geometry_name, name, name2)

    def _init_from_info(self, i) -> None:
        LOG.debug(f"{i}")
        self.cube_id = i["id"]
        self.geometry_id = i["geometry_id"]
        self.geometry_name = i["geometry_name"]
        self.origin = i["origin"]
        self.v_i = i["d_inline"]
        self.v_x = i["d_xline"]
        self.n_i = i["max_inline"]+1
        self.n_x = i["max_xline"]+1
        self.n_samples = i["nz"]
        self.time_step = i["z_step"]
        self.domain = i["domain"]
        self.data_start = i["z_start"]
        self.min_i = i["min_inline"]
        self.min_x = i["min_xline"]

    def _create(self, job_id: Optional[int] = None):
        # url = f"{self.server_url}/seismic_3d/create/{self.project_id}/"
        url = f"{self.server_url}/seismic_3d/geometry/new_cube/{self.geometry_id}/"
        res_status = 200
        res_id = -1
        try:
            cube_out = self._get_info()
            del cube_out["id"]
            del cube_out["geometry_id"]
            del cube_out["geometry_name"]
            LOG.info(f"{cube_out=}")
            body = json.dumps(cube_out).encode("utf8")
            with requests.post(
                    url, data=body, params={"job_id": job_id}, headers={"Content-Type": "application/json", "x-di-authorization": self.token}
                ) as resp:
                if resp.status_code != 200:
                    LOG.error("Failed to create cube, response code %s", resp.status_code)
                    raise RuntimeError(f"Failed to create cube, response code {resp.status_code}")
                resp_j = json.loads(resp.content)
                LOG.debug("Reply: %s", resp_j)
                self.cube_id = resp_j["id"]
        except requests.exceptions.ConnectionError as ex:
            LOG.error("Exception during POST: %s", str(ex))
            raise ex


    def write_fragment(self, inline_no: int, xline_no: int, data_array):
        url = f"{self.server_url}/seismic_3d/data/rect_fragment/{self.cube_id}/?inline_no={inline_no}&xline_no={xline_no}"
        ninl, ncdps, nz = data_array.shape
        pref = struct.pack('<iii', nz, ncdps, ninl)
        data = pref + data_array.tobytes()
        res_status = 200
        with requests.post(url, data=data, headers={"Content-Type": "application/octet-stream"}) as resp:
            res_status = resp.status_code
            if resp.status_code != 200:
                LOG.error("Failed to store fragment to cube, response code %s", resp.status_code)
                return res_status
        

    def write_fragment_z(self, inline_no: int, xline_no: int, z_no: int, data_array):
        url = f"{self.server_url}/seismic/data/rect_fragment_z/{self.cube_id}/?inline_no={inline_no}&xline_no={xline_no}&z_no={z_no}"
        ninl, ncdps, nz = data_array.shape
        pref = struct.pack('<iii', nz, ncdps, ninl)
        data = pref + data_array.tobytes()
        res_status = 200
        with requests.post(url, data=data, headers={"Content-Type": "application/octet-stream"}) as resp:
            res_status = resp.status_code
            if resp.status_code != 200:
                LOG.error("Failed to store fragment to cube, response code %s", resp.status_code)
                return res_status
