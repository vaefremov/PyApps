import logging
import numpy as np
import requests
import json
import struct
from typing import Optional

LOG = logging.getLogger(__name__)

SAMPLE_BYTE_LEN = 4 # Corresponds to lsb format of data
MAXFLOAT = 3.40282347e+38 ## stands for undefined values of parameters
MAXFLOAT09 = 0.9*3.40282347e+38 ## stands for undefined values of parameters


class DIHorizon3D:
    def __init__(self, project_id: int, geometry_name: str, name: str) -> None:
        self.server_url = ""
        self.token = ""
        self.origin = None
        self.v_i = None
        self.v_x = None
        self.n_i = -1
        self.n_x = -1
        self.norm_v_i = None
        self.norm_v_x = None
        self.domain = "No domain"
        self.mode = "None"
        self.project_id = project_id
        self.geometry_name = geometry_name
        self.name = name
        self.geometry_id = None
        self.horizon_id = None
        self.n_layers = 0
        
    def __repr__(self):
        return f"DIHorizon3D: {self.horizon_id=} {self.geometry_name=} {self.name=}"

    def _read_info(self) -> None:
        with requests.get(f"{self.server_url}/horizons/3d/list/{self.project_id}/") as resp:
            if resp.status_code != 200:
                LOG.error("Cant' get list of cubes: %s", resp.status_code)
                return None
            resp_j = json.loads(resp.content)
            for i in resp_j:
                if (i["geometry_name"] == self.geometry_name) and (i["name"] == self.name):
                    self.horizon_id = i["id"]
                    self.geometry_id = i["geometry_id"]
                    LOG.debug(f"{i}")
                    self.origin = i["origin"]
                    self.v_i = i["dx"]
                    self.v_x = i["dy"]
                    self.n_i = i["nx"]
                    self.n_x = i["ny"]
                    self.domain = i["domain"]
                    self.mode = i["mode"]
            if self.horizon_id is None:
                raise RuntimeError(f"Horizon 3d {self.geometry_name}/{self.name} not found in {self.project_id=}")

    def _get_info(self):
        i = {
            "geometry_name": self.geometry_name,
            "geometry_id": self.geometry_id,
            "name": self.name,
            "nx": self.n_i,
            "ny": self.n_x,
            "origin": self.origin,
            "dx": self.v_i,
            "dy": self.v_x,
            "domain": self.domain,
            "id": self.horizon_id,
            "mode": self.mode,
            "domain": self.domain
        }
        return i

    def get_data(self) -> Optional[np.ndarray]:
        url = f"{self.server_url}/horizons/3d/data/{self.project_id}/{self.horizon_id}/"
        with requests.get(url) as resp:
            bytes_read = len(resp.content)
            raw_data = resp.content
            if resp.status_code != 200:
                LOG.error("Request finished with error: %s", resp.status_code)
            nx, ny = struct.unpack("<ii", raw_data[:8])
            LOG.debug(f"{nx=}, {ny=}")
            gr_arr = np.frombuffer(raw_data[8:], dtype=np.float32)
            gr_arr.shape = (nx, ny)
            return gr_arr
