import logging
from optparse import Option
import numpy as np
import requests
import json
from typing import Optional, List, Tuple
from pydantic import BaseModel, Field, validator

LOG = logging.getLogger(__name__)

MAXFLOAT = 3.40282347e+38 ## stands for undefined values of parameters


LOG = logging.getLogger(__name__)

class AreaInfo(BaseModel):
    name: str
    id: Optional[int] = None
    area: List[Tuple[float, float]]
    ts: str
    owner: Optional[str]

    @validator("area", pre=True)
    def ensure_list(cls, value):
        if value is None:
            return []
        return value

class DIArea:
    def __init__(self, project_id: int, name: str) -> None:
        self.server_url = ""
        self.token = ""
        self.project_id = project_id
        self.name = name
        self._area_info: AreaInfo = AreaInfo(name=self.name, id=None, area=[], ts="", owner=None)

    @property
    def area_info(self) -> AreaInfo:
        if self._area_info.id is None:
            self._area_info = self._read_info()
        return self._area_info

    @property
    def polygon(self) -> List[Tuple[float, float]]:
        if self._area_info.id is None:
            self._area_info = self._read_info()
        return self.area_info.area

    @polygon.setter
    def polygon(self, val: List[Tuple[float, float]]):
        self._area_info.area = val
        self._update()

    def __repr__(self):
        return f"DIArea: {self._area_info.id=} {self.name=}"

    def _read_info(self) -> AreaInfo:
        url = f"{self.server_url}/grids/areas/list/{self.project_id}/"
        area_id = None
        with requests.get(url) as resp:
            if resp.status_code != 200:
                LOG.error(f"Can't get list of areas from {self.server_url} ({resp.status_code}): {resp.content}")
                raise RuntimeError(f"Can't get list of areas from {self.server_url} ({resp.status_code}): {resp.content}")

            resp_json = resp.json()            
            for item in resp_json:
                if item["name"] == self.name:
                    area_id = item["id"]
                    LOG.debug(f"Area info: {self._area_info}")
                    break
            if area_id is None:
                raise RuntimeError(f"Area {self.name} not found in project {self.project_id}")

        url = f"{self.server_url}/grids/areas/properties/{self.project_id}/{area_id}/"
        with requests.get(url) as resp:
            if resp.status_code != 200:
                LOG.error(f"Unable to read area properties ({resp.status_code=}): {resp.content}")
                raise RuntimeError(f"Unable to read area properties ({resp.status_code=}): {resp.content}")
            return AreaInfo(**resp.json())

    def _get_info(self) -> dict:
        if self._area_info is None:
            self._read_info()
        return self.area_info.dict()

    def _update(self) -> None:
        url = f"{self.server_url}/grids/areas/update/{self.project_id}/"
        res_status = 200
        res_id = -1
        try:
            hor_out = self._get_info()
            del hor_out["id"]
            del hor_out["ts"]
            del hor_out["owner"]
            LOG.info(f"{hor_out=}")
            body = json.dumps(hor_out).encode("utf8")
            with requests.post(
                    url, data=body, headers={"Content-Type": "application/json", "x-di-authorization": self.token}
                ) as resp:
                if resp.status_code != 200:
                    LOG.error("Failed to create area, response code %s", resp.status_code)
                    raise RuntimeError(f"Failed to update area on {self.server_url}, response code {resp.status_code}, {resp.content=}")
                resp_j = json.loads(resp.content)
                LOG.debug("Reply: %s", resp_j)
                self._area_info.id = resp_j["id"]
                self._area_info.ts = resp_j["ts"]
        except requests.exceptions.ConnectionError as ex:
            LOG.error("Exception during POST: %s", str(ex))
            raise ex

    def _create(self) -> None:
        url = f"{self.server_url}/grids/areas/update/{self.project_id}/"
        res_status = 200
        res_id = -1
        try:
            hor_out = self._area_info.dict()
            del hor_out["id"]
            del hor_out["ts"]
            del hor_out["owner"]
            LOG.info(f"{hor_out=}")
            body = json.dumps(hor_out).encode("utf8")
            with requests.post(
                    url, data=body, headers={"Content-Type": "application/json", "x-di-authorization": self.token}
                ) as resp:
                if resp.status_code != 200:
                    LOG.error("Failed to create area, response code %s", resp.status_code)
                    raise RuntimeError(f"Failed to update area on {self.server_url}, response code {resp.status_code}, {resp.content=}")
                resp_j = json.loads(resp.content)
                LOG.debug("Reply: %s", resp_j)
                self._area_info.id = resp_j["id"]
                self._area_info.ts = resp_j["ts"]
        except requests.exceptions.ConnectionError as ex:
            LOG.error("Exception during POST: %s", str(ex))
            raise ex


def new_area(server_url: str, token: str, project_id: int, name: str, path: List[Tuple[float, float]]) -> DIArea:
    area = DIArea(project_id, name)
    area.server_url = server_url
    area.token = token
    area._area_info = AreaInfo(name=name, id=None, area=path, ts="", owner=None)
    area._create()
    return area

