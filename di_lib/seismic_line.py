import logging
import numpy as np
import requests
import json
import struct

LOG = logging.getLogger(__name__)

SAMPLE_BYTE_LEN = 4 # Corresponds to lsb format of data
MAXFLOAT = 3.40282347e+38 ## stands for undefined values of parameters
MAXFLOAT09 = 0.9*3.40282347e+38 ## stands for undefined values of parameters

class DISeismicLine:
    def __init__(self, project_id: int, name: str, name2: str) -> None:
        self.project_id = project_id
        self.server_url = ""
        self.token = ""
        self.n_samples = -1
        self.time_step = 0
        self.domain = None
        self.data_start = None
        self.name = name
        self.name2 = name2
        self.line_id = None
        self.geometry = []
        self.cdps = []
        
    def __repr__(self):
        return f"DISeismicLine: {self.line_id} {self.name} {self.name2}"

    def _read_info(self) -> None:
        with requests.get(f"{self.server_url}/seismic_2d/list/{self.project_id}/") as resp:
            if resp.status_code != 200:
                LOG.error("Cant' get list of lines: %s", resp.status_code)
                return None
            resp_j = json.loads(resp.content)
            for i in resp_j:
                if (i["name"] == self.name) and (i["name2"] == self.name2):
                    self.line_id = i["id"]
                    LOG.debug(f"{i}")
                    self.n_samples = i["nz"]
                    self.time_step = i["z_step"]
                    self.domain = i["domain"]
                    self.data_start = i["z_start"]
            if self.line_id is None:
                raise RuntimeError(f"Line {self.name}/{self.name2} not found in {self.project_id=}")
        # read geometry
        body = json.dumps([self.line_id]).encode()
        with requests.post(f"{self.server_url}/seismic_2d/list_geom/{self.project_id}/", data=body, headers={"Content-Type": "application/json", "x-di-authorization": self.token}) as resp:
            if resp.status_code != 200:
                LOG.error("Cant' get geometry of line: %s", resp.status_code)
                return None
            resp_j = json.loads(resp.content)
            self.geometry = resp_j[0]["geometry"]
            self.cdps = resp_j[0]["cdps"]

    def _get_info(self):
        i = {
            "name": self.name,
            "name2": self.name2,
            "nz": self.n_samples,
            "domain": self.domain,
            "z_start": self.data_start,
            "z_step": self.time_step,
            "id": self.line_id,
            "geometry": self.geometry,
            "cdps": self.cdps
        }
        return i

    def get_data(self) -> np.ndarray:
        url = f"{self.server_url}/seismic_2d/data/{self.line_id}/"
        with requests.get(url) as resp:
            bytes_read = len(resp.content)
            raw_data = resp.content
            if resp.status_code != 200:
                LOG.error("Request finished with error: %s", resp.status_code)
                raise RuntimeError(f"Request failed: {resp.status_code}")
            nz, ncdps = struct.unpack("<ii", raw_data[:8])
            LOG.debug(f"{nz=}, {ncdps=}")
            gr_arr = np.frombuffer(raw_data[8:], dtype=np.float32)
            gr_arr.shape = (ncdps, nz)
            return gr_arr

class DISeismicLineWriter(DISeismicLine):
    def __init__(self, project_id: int, name: str, name2: str) -> None:
        super().__init__(project_id, name, name2)

    def _init_from_info(self, i) -> None:
        LOG.debug(f"{i}")
        self.n_samples = i["nz"]
        self.time_step = i["z_step"]
        self.domain = i["domain"]
        self.data_start = i["z_start"]
        self.geometry = i["geometry"]
        self.cdps = i["cdps"]

    def _create(self):
        url = f"{self.server_url}/seismic_2d/create/{self.project_id}/"
        res_status = 200
        res_id = -1
        try:
            line_out = self._get_info()
            del line_out["id"]
            LOG.info(f"{line_out=}")
            body = json.dumps(line_out).encode("utf8")
            with requests.post(
                    url, data=body, headers={"Content-Type": "application/json", "x-di-authorization": self.token}
                ) as resp:
                if resp.status_code != 200:
                    LOG.error("Failed to create line, response code %s, %s", resp.status_code, resp.content)
                    raise RuntimeError(f"Failed to create line, response code {resp.status_code}")
                resp_j = json.loads(resp.content)
                LOG.info("Reply: %s", resp_j)
                self.line_id = resp_j["id"]
        except requests.exceptions.ConnectionError as ex:
            LOG.error("Exception during POST: %s", str(ex))
            raise ex

    def write_data(self, data_array: np.ndarray):
        url = f"{self.server_url}/seismic_2d/data/{self.line_id}/"
        ncdps, nz = data_array.shape
        pref = struct.pack('<ii', nz, ncdps)
        data = pref + data_array.tobytes()
        res_status = 200
        with requests.post(url, data=data, headers={"Content-Type": "application/octet-stream"}) as resp:
            res_status = resp.status_code
            if resp.status_code != 200:
                LOG.error("Failed to store line data, response code %s", resp.status_code)
                return res_status

