import logging
import numpy as np
import requests
import json
import struct
from typing import Optional, List

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
        self._layers_names = None

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
                    self.min_i = i["min_nx"]
                    self.min_x = i["min_ny"]
                    self.domain = i["domain"]
                    self.mode = i["mode"]
                    self.n_layers = i["n_layers"]
                    self._layers_names = i["layers_names"]
            if self.horizon_id is None:
                raise RuntimeError(f"Horizon 3d {self.geometry_name}/{self.name} not found in {self.project_id=}")

    def _get_info(self):
        i = {
            "geometry_name": self.geometry_name,
            "geometry_id": self.geometry_id,
            "name": self.name,
            "nx": self.n_i,
            "ny": self.n_x,
            "min_nx": self.min_i,
            "min_ny": self.min_x,
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

class DIHorizon3DWriter(DIHorizon3D):
    def __init__(self, project_id: int, geometry_name, name: str) -> None:
        super().__init__(project_id, geometry_name, name)

    def _init_from_info(self, i) -> None:
        LOG.debug(f"{i}")
        self.horizon_id = i["id"]
        self.geometry_id = i["geometry_id"]
        self.geometry_name = i["geometry_name"]
        self.origin = i["origin"]
        self.v_i = i["dx"]
        self.v_x = i["dy"]
        self.n_i = i["nx"]
        self.n_x = i["ny"]
        self.min_i = i["min_nx"]
        self.min_x = i["min_ny"]
        self.domain = i["domain"]
        self.mode = i["mode"]

    def _create(self) -> None:
        url = f"{self.server_url}/horizons/3d/create_empty/{self.project_id}/?geometry_id={self.geometry_id}"
        res_status = 200
        res_id = -1
        try:
            hor_out = self._get_info()
            del hor_out["id"]
            del hor_out["geometry_id"]
            del hor_out["geometry_name"]
            LOG.info(f"{hor_out=}")
            body = json.dumps(hor_out).encode("utf8")
            with requests.post(
                    url, data=body, headers={"Content-Type": "application/json", "x-di-authorization": self.token}
                ) as resp:
                if resp.status_code != 200:
                    LOG.error("Failed to create horizon, response code %s", resp.status_code)
                    raise RuntimeError(f"Failed to create horizon, response code {resp.status_code}")
                resp_j = json.loads(resp.content)
                LOG.debug("Reply: %s", resp_j)
                self.horizon_id = resp_j["id"]
        except requests.exceptions.ConnectionError as ex:
            LOG.error("Exception during POST: %s", str(ex))
            raise ex

    def write_data(self, data_array):
        if data_array.dtype != np.float32:
            raise ValueError('Data type must be float32')
        url = f"{self.server_url}/horizons/3d/update_data/{self.project_id}/{self.horizon_id}/"
        nx, ny = data_array.shape
        pref = struct.pack('<ii', nx, ny)
        data = pref + data_array.tobytes()
        res_status = 200
        with requests.post(url, data=data, headers={"Content-Type": "application/octet-stream", "x-di-authorization": self.token}) as resp:
            res_status = resp.status_code
            if resp.status_code != 200:
                LOG.error("Failed to store horizon data, response code %s", resp.status_code)
                return res_status

    def write_data_fragment(self, start_nx: int, start_ny: int, data_array):
        if data_array.dtype != np.float32:
            raise ValueError('Data type must be float32')
        url = f"{self.server_url}/horizons/3d/update_data_fragment/{self.project_id}/{self.horizon_id}/"
        nx, ny = data_array.shape
        pref = struct.pack('<ii', nx, ny)
        data = pref + data_array.tobytes()
        res_status = 200
        with requests.post(url, data=data, params={"start_nx": start_nx, "start_ny": start_ny},
                           headers={"Content-Type": "application/octet-stream", "x-di-authorization": self.token}) as resp:
            res_status = resp.status_code
            if resp.status_code != 200:
                LOG.error("Failed to store horizon data, response code %s", resp.status_code)
                return res_status

class DIAttribute2D(DIHorizon3DWriter):
    """Reader/writer for layered attributes.
    """
    def __init__(self, project_id: int, geometry_name, name: str, name2: str) -> None:
        super().__init__(project_id, geometry_name, name)
        self.name2 = name2

    def __repr__(self):
        return f"DIAttribute3D: {self.horizon_id=} {self.geometry_name=} {self.name=} {self.name2=}"
    
    def _read_info(self) -> None:
        with requests.get(f"{self.server_url}/attributes/3d/list/{self.project_id}/") as resp:
            if resp.status_code != 200:
                LOG.error("Cant' get list of attributes: %s", resp.status_code)
                return None
            resp_j = json.loads(resp.content)
            for i in resp_j:
                if (i["geometry_name"] == self.geometry_name) and (i["name"] == self.name) and (i["name2"] == self.name2):
                    self.horizon_id = i["id"]
                    self.geometry_id = i["geometry_id"]
                    LOG.debug(f"{i}")
                    self.origin = i["origin"]
                    self.v_i = i["dx"]
                    self.v_x = i["dy"]
                    self.n_i = i["nx"]
                    self.n_x = i["ny"]
                    self.min_i = i["min_nx"]
                    self.min_x = i["min_ny"]
                    self.domain = i["domain"]
                    self.mode = i["mode"]
                    self.n_layers = i["n_layers"]
                    self._layers_names = i["layers_names"]
                    self.name2 = i["name2"]
            if self.horizon_id is None:
                raise RuntimeError(f"Attribute 2d {self.geometry_name}/{self.name} not found in {self.project_id=}")

    def _get_info(self):
        i = {
            "geometry_name": self.geometry_name,
            "geometry_id": self.geometry_id,
            "name": self.name,
            "name2": self.name2,
            "nx": self.n_i,
            "ny": self.n_x,
            "min_nx": self.min_i,
            "min_ny": self.min_x,
            "origin": self.origin,
            "dx": self.v_i,
            "dy": self.v_x,
            "domain": self.domain,
            "id": self.horizon_id,
            "mode": self.mode,
            "domain": self.domain
        }
        return i

    def _create(self) -> None:
        url = f"{self.server_url}/attributes/3d/create_empty/{self.project_id}/?geometry_id={self.geometry_id}"
        res_status = 200
        res_id = -1
        try:
            hor_out = self._get_info()
            del hor_out["id"]
            del hor_out["geometry_id"]
            del hor_out["geometry_name"]
            LOG.info(f"{hor_out=}")
            body = json.dumps(hor_out).encode("utf8")
            with requests.post(
                    url, data=body, headers={"Content-Type": "application/json", "x-di-authorization": self.token}
                ) as resp:
                if resp.status_code != 200:
                    LOG.error("Failed to create horizon, response code %s", resp.status_code)
                    raise RuntimeError(f"Failed to create horizon, response code {resp.status_code}")
                resp_j = json.loads(resp.content)
                LOG.debug("Reply: %s", resp_j)
                self.horizon_id = resp_j["id"]
                self._layers_names = resp_j["layers_names"]
        except requests.exceptions.ConnectionError as ex:
            LOG.error("Exception during POST: %s", str(ex))
            raise ex

    def get_all_data(self) -> Optional[np.ndarray]:
        """Reads layered data. Returns array with shape (nlayers, nx, ny).
        """
        url = f"{self.server_url}/attributes/3d/layers_data/{self.project_id}/{self.horizon_id}/"
        with requests.get(url) as resp:
            bytes_read = len(resp.content)
            raw_data = resp.content
            if resp.status_code != 200:
                LOG.error("Request finished with error: %s", resp.status_code)
            nlayers, nx, ny = struct.unpack("<iii", raw_data[:12])
            LOG.debug(f"{nlayers=} {nx=}, {ny=}")
            gr_arr = np.frombuffer(raw_data[12:], dtype=np.float32)
            gr_arr.shape = (nlayers, nx, ny)
            return gr_arr

    def get_data(self) -> Optional[np.ndarray]:
        """Reads attribute data (corresponds to layer 1). Returns array with shape (nx, ny).
        """
        url = f"{self.server_url}/attributes/3d/data/{self.project_id}/{self.horizon_id}/"
        with requests.get(url, params={"layer_name": self._layers_names[1]}) as resp:
            bytes_read = len(resp.content)
            raw_data = resp.content
            if resp.status_code != 200:
                LOG.error("Request finished with error: %s", resp.status_code)
            nx, ny = struct.unpack("<ii", raw_data[:8])
            LOG.debug(f"{nx=}, {ny=}")
            gr_arr = np.frombuffer(raw_data[8:], dtype=np.float32)
            gr_arr.shape = (nx, ny)
            return gr_arr

    def write_data(self, data_array):
        # make sure type of the input data is float32
        if data_array.dtype != np.float32:
            raise ValueError('Data type must be float32')
        url = f"{self.server_url}/attributes/3d/update_attribute_data/{self.project_id}/{self.horizon_id}/"
        nx, ny = data_array.shape
        pref = struct.pack('<ii', nx, ny)
        data = pref + data_array.tobytes()
        res_status = 200
        with requests.post(url, data=data, headers={"Content-Type": "application/octet-stream", "x-di-authorization": self.token}) as resp:
            res_status = resp.status_code
            if resp.status_code != 200:
                LOG.error(f"Failed to store horizon data, response code {resp.status_code}, {resp.content}")
                return res_status
            
    def write_horizon_data(self, data_array):
        # make sure type of the input data is float32
        if data_array.dtype != np.float32:
            raise ValueError('Data type must be float32')
        url = f"{self.server_url}/attributes/3d/update_horizon_data/{self.project_id}/{self.horizon_id}/"
        nx, ny = data_array.shape
        pref = struct.pack('<ii', nx, ny)
        data = pref + data_array.tobytes()
        res_status = 200
        with requests.post(url, data=data, headers={"Content-Type": "application/octet-stream", "x-di-authorization": self.token}) as resp:
            res_status = resp.status_code
            if resp.status_code != 200:
                LOG.error(f"Failed to store horizon data, response code {resp.status_code}, {resp.content}")
                return res_status
            
    def write_all_data(self, data_array, layers_names, **kwargs):
        # make sure type of the input data is float32
        if data_array.dtype != np.float32:
            raise ValueError('Data type must be float32')
        url = f"{self.server_url}/attributes/3d/update_attribute_entire_data/{self.project_id}/{self.horizon_id}/"
        nlayers, nx, ny = data_array.shape
        if len(layers_names) != nlayers:
            raise ValueError(f'Number of layers names must be equal to layers number ({len(layers_names)} != {nlayers})')
        self._layers_names = layers_names
        pref = struct.pack('<iii', nlayers, nx, ny)
        data = pref + data_array.tobytes()
        res_status = 200
        params={"lnm": self._layers_names,
                "min_nx": kwargs.get("min_nx"),
                "min_ny": kwargs.get("min_ny")
                }
        with requests.post(url, data=data, params=params, headers={"Content-Type": "application/octet-stream", "x-di-authorization": self.token}) as resp:
            res_status = resp.status_code
            if resp.status_code != 200:
                LOG.error(f"Failed to store horizon data, response code {resp.status_code}, {resp.content}")
                return res_status

    @property
    def layers_names(self):
        return self._layers_names
