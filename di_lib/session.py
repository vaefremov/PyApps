import requests
import json
import sys

from .seismic_cube import DISeismicCube, DISeismicCubeWriter
from .seismic_line import DISeismicLine, DISeismicLineWriter
from .attribute import DIHorizon3D
import logging

LOG = logging.getLogger(__name__)

class DISession:
    """Effectively, implements the Factory pattern to
    produce data object in the right context.
    """
    def __init__(self, project_id: int, server_url: str, token: str) -> None:
        self.project_id = project_id
        self.server_url = server_url
        self.token = token

    def ping(self):
        with requests.get(f"{self.server_url}/service/ping/") as resp:
            if resp.status_code != 200:
                LOG.error("Ping failed: %s", resp.status_code)
                raise RuntimeError(f"Ping failed: {resp.status_code=}")
            resp_j = json.loads(resp.content)
            return resp_j


    def list_cubes(self):
        with requests.get(f"{self.server_url}/seismic_3d/list/{self.project_id}/") as resp:
            if resp.status_code != 200:
                LOG.error("Cant' get list of cubes: %s", resp.status_code)
                raise RuntimeError(f"Cant' get list of cubes: {resp.status_code=}")
                return None
            resp_j = json.loads(resp.content)
            return resp_j


    def get_cube(self, geometry_name: str, name: str, attr_name: str) -> DISeismicCube:
        cube = DISeismicCube(self.project_id, geometry_name, name, attr_name)
        cube.server_url = self.server_url
        cube.token = self.token
        cube._read_info()
        return cube

    def create_cube_writer_as_other(self, original_cube: DISeismicCube, name: str, attr_name: str, **kw) -> DISeismicCubeWriter:
        cube_writer = DISeismicCubeWriter(self.project_id, name, attr_name)
        cube_writer.server_url = self.server_url
        cube_writer.token = self.token
        original_info = original_cube._get_info()
        new_info = {}
        new_info.update(original_info)
        new_info["z_step"] = kw.get("z_step", original_info["z_step"])
        new_info["z_start"] = kw.get("z_start", original_info["z_start"])
        new_info["domain"] = kw.get("domain", original_info["domain"])
        new_nz = kw.get("nz", None)
        if new_nz is None:
            # Recalculate nz according to new z_step
            new_nz = round(original_info["nz"] * (original_info["z_step"]  / new_info["z_step"] ))
        new_info["nz"] = new_nz
        cube_writer._init_from_info(new_info)
        cube_writer._create()
        return cube_writer

    def delete_cube_by_id(self, cube_id: int):
        with requests.delete(f'http://localhost:9990/seismic_3d/delete/{cube_id}/', headers={"Content-Type": "application/json", "x-di-authorization": self.token}) as resp:
            if resp.status_code != 200:
                LOG.error("Delete failed: %s  / %s", resp.status_code, resp.content)
                raise RuntimeError(f"Delete failed: {resp.status_code=}")

    def list_lines(self):
        with requests.get(f"{self.server_url}/seismic_2d/list/{self.project_id}/") as resp:
            if resp.status_code != 200:
                LOG.error("Cant' get list of lines: %s", resp.status_code)
                raise RuntimeError(f"Cant' get list of lines: {resp.status_code=}")
                return None
            resp_j = json.loads(resp.content)
            return resp_j


    def get_line(self, geometry_name: str, name: str, attr_name: str) -> DISeismicLine:
        line = DISeismicLine(self.project_id, geometry_name, name, attr_name)
        line.server_url = self.server_url
        line.token = self.token
        line._read_info()
        return line

    def delete_line_by_id(self, line_id: int):
        with requests.delete(f'http://localhost:9990/seismic_2d/delete/{line_id}/', headers={"Content-Type": "application/json", "x-di-authorization": self.token}) as resp:
            if resp.status_code != 200:
                LOG.error("Delete failed: %s  / %s", resp.status_code, resp.content)
                raise RuntimeError(f"Delete failed: {resp.status_code=}")

    def create_line_writer_as_other(self, original_line: DISeismicLine, name: str, attr_name: str, **kw) -> DISeismicLineWriter:
        line_writer = DISeismicLineWriter(self.project_id, name, attr_name)
        line_writer.server_url = self.server_url
        line_writer.token = self.token
        # TODO: Here we should copy descriptive information to the new cube
        original_info = original_line._get_info()
        new_info = {}
        new_info.update(original_info)
        new_info["z_step"] = kw.get("z_step", original_info["z_step"])
        new_info["z_start"] = kw.get("z_start", original_info["z_start"])
        new_info["domain"] = kw.get("domain", original_info["domain"])
        new_nz = kw.get("nz", None)
        if new_nz is None:
            # Recalculate nz according to new z_step
            new_nz = round(original_info["nz"] * (original_info["z_step"]  / new_info["z_step"] ))
        new_info["nz"] = new_nz
        line_writer._init_from_info(new_info)
        # and create cube on server
        line_writer._create()
        return line_writer


    def list_attributes_3d(self):
        with requests.get(f"{self.server_url}/horizons/3d/list/{self.project_id}/") as resp:
            if resp.status_code != 200:
                LOG.error("Cant' get list of 3d attributes: %s", resp.status_code)
                raise RuntimeError(f"Cant' get list of 3d attributes: {resp.status_code=}")
                return None
            resp_j = json.loads(resp.content)
            return resp_j

    def get_attribute_3d(self, geometry_name: str, name: str) -> DIHorizon3D:
        hor = DIHorizon3D(self.project_id, geometry_name, name)
        hor.server_url = self.server_url
        hor.token = self.token
        hor._read_info()
        return hor

    def create_attribute_3d_as_other(self, name: str):
        raise NotImplementedError("Not implemented!")
    