import requests
import json
import sys

from .seismic_cube import DISeismicCube, DISeismicCubeWriter, DIGeometry
from .seismic_line import DISeismicLine, DISeismicLineWriter
from .attribute import DIAttribute2D, DIHorizon3D, DIHorizon3DWriter
from .di_job import DiJob
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

    def list_geometries(self):
        with requests.get(f"{self.server_url}/seismic_3d/geometries/{self.project_id}/") as resp:
            if resp.status_code != 200:
                LOG.error("Cant' get list of geometries: %s", resp.status_code)
                raise RuntimeError(f"Cant' get list of geometries: {resp.status_code=}")
            resp_j = json.loads(resp.content)
            return resp_j

    def get_geometry(self, name: str) -> DIHorizon3D:
        geometry = DIGeometry(self.project_id, name)
        geometry.server_url = self.server_url
        geometry.token = self.token
        geometry._read_info()
        return geometry
    
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
        job_id = kw.get("job_id", None)
        if new_nz is None:
            # Recalculate nz according to new z_step
            new_nz = round(original_info["nz"] * (original_info["z_step"]  / new_info["z_step"] ))
        new_info["nz"] = new_nz
        cube_writer._init_from_info(new_info)
        cube_writer._create(job_id)
        return cube_writer

    def create_cube_writer_in_geometry(self, geometry_name: str, name: str, attr_name: str, **kw) -> DISeismicCubeWriter:
        raise NotImplementedError()
        # Find geometry with geometry_name
        geometry = self.get_geometry(geometry_name)
        cube_writer = DISeismicCubeWriter(self.project_id, name, attr_name)
        cube_writer.server_url = self.server_url
        cube_writer.token = self.token
        new_info = {}
        new_info["z_step"] = kw.get("z_step", 1)
        new_info["z_start"] = kw.get("z_start", 0)
        new_info["domain"] = kw.get("domain", "")
        new_nz = kw.get("nz", None)
        job_id = kw.get("job_id", None)
        if new_nz is None:
            # Recalculate nz according to new z_step
            new_nz = round(original_info["nz"] * (original_info["z_step"]  / new_info["z_step"] ))
        new_info["nz"] = new_nz
        cube_writer._init_from_info(new_info)
        cube_writer._create(job_id)
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


    def list_attributes_2d(self):
        with requests.get(f"{self.server_url}/attributes/3d/list/{self.project_id}/") as resp:
            if resp.status_code != 200:
                LOG.error("Cant' get list of 3d attributes: %s", resp.status_code)
                raise RuntimeError(f"Cant' get list of 3d attributes: {resp.status_code=}")
                return None
            resp_j = json.loads(resp.content)
            return [a for a in resp_j if (a["n_layers"] is not None) and (a["n_layers"]>1)]

    def list_horizons_3d(self):
        with requests.get(f"{self.server_url}/horizons/3d/list/{self.project_id}/") as resp:
            if resp.status_code != 200:
                LOG.error("Cant' get list of 3d attributes: %s", resp.status_code)
                raise RuntimeError(f"Cant' get list of 3d attributes: {resp.status_code=}")
                return None
            resp_j = json.loads(resp.content)
            return [a for a in resp_j if (a["n_layers"] is None) or (a["n_layers"]==1)]

    def get_horizon_3d(self, geometry_name: str, name: str) -> DIHorizon3D:
        hor = DIHorizon3D(self.project_id, geometry_name, name)
        hor.server_url = self.server_url
        hor.token = self.token
        hor._read_info()
        return hor

    def get_horizon_3d_writer(self, geometry_name: str, name: str) -> DIHorizon3DWriter:
        hor = DIHorizon3DWriter(self.project_id, geometry_name, name)
        hor.server_url = self.server_url
        hor.token = self.token
        hor._read_info()
        return hor

    def create_horizon_3d_writer_as_other(self, original_attribute: DIHorizon3D, name: str, **kw):
        attr_writer = DIHorizon3DWriter(self.project_id, original_attribute.geometry_name, name)
        attr_writer.server_url = self.server_url
        attr_writer.token = self.token
        original_info = original_attribute._get_info()
        new_info = {}
        new_info.update(original_info)
        new_info["domain"] = kw.get("domain", original_info["domain"])
        new_info["mode"] = kw.get("mode", original_info["mode"])
        new_info["min_nx"] = kw.get("min_nx", original_info["min_nx"])
        new_info["min_ny"] = kw.get("min_ny", original_info["min_ny"])
        attr_writer._init_from_info(new_info)
        attr_writer._create()
        return attr_writer

    def get_attribute_2d_writer(self, geometry_name: str, name: str, name2: str) -> DIAttribute2D:
        hor = DIAttribute2D(self.project_id, geometry_name, name, name2)
        hor.server_url = self.server_url
        hor.token = self.token
        hor._read_info()
        return hor

    def create_attribute_2d_writer_as_other(self, original_attribute: DIHorizon3D, name: str, name2: str, copy_horizon_data: bool=False, **kw) -> DIAttribute2D:
        attr_writer = DIAttribute2D(self.project_id, original_attribute.geometry_name, name, name2)
        attr_writer.server_url = self.server_url
        attr_writer.token = self.token
        original_info = original_attribute._get_info()
        new_info = {}
        new_info.update(original_info)
        new_info["domain"] = kw.get("domain", original_info["domain"])
        new_info["mode"] = kw.get("mode", original_info["mode"])
        new_info["min_nx"] = kw.get("min_nx", original_info["min_nx"])
        new_info["min_ny"] = kw.get("min_ny", original_info["min_ny"])
        attr_writer._init_from_info(new_info)
        attr_writer._create()
        if copy_horizon_data:
            hor_dt = original_attribute.get_data()
            attr_writer.write_horizon_data(hor_dt)
        return attr_writer
        
    def create_attribute_2d_writer_for_cube(self,  original_cube: DISeismicCube, name: str, name2: str, **kw) -> DIAttribute2D:
        attr_writer = DIAttribute2D(self.project_id, original_cube.geometry_name, name, name2)
        attr_writer.server_url = self.server_url
        attr_writer.token = self.token
        original_info = original_cube._get_info()
        new_info = {
            "id": None,
            "dx": original_info["d_inline"],
            "dy": original_info["d_xline"],
            "origin": original_info["origin"],
            "nx": original_info["max_inline"], # we do not add 1 to compensate for in/x-line numbers starting from 1
            "ny": original_info["max_xline"],
            "geometry_id": original_info["geometry_id"],
            "geometry_name": original_info["geometry_name"]
        }
        # new_info.update(original_info)
        new_info["domain"] = kw.get("domain", original_info["domain"])
        new_info["mode"] = None
        new_info["min_nx"] = kw.get("min_nx", original_info["min_inline"])
        new_info["min_ny"] = kw.get("min_ny", original_info["min_xline"])
        attr_writer._init_from_info(new_info)
        attr_writer._create()
        return attr_writer

    def delete_attribute_by_id(self, attr_id: int):
        with requests.delete(f'{self.server_url}/attributes/3d/delete/{attr_id}/', headers={"Content-Type": "application/json", "x-di-authorization": self.token}) as resp:
            if resp.status_code != 200:
                LOG.error("Delete failed: %s  / %s", resp.status_code, resp.content)
                raise RuntimeError(f"Delete failed: {resp.status_code=}")

    def get_job_by_id(self, job_id):
        job = DiJob(job_id)
        job.server_url = self.server_url
        return job

    def list_areas(self):
        with requests.get(f"{self.server_url}/grids/areas/list/{self.project_id}/") as resp:
            if resp.status_code != 200:
                LOG.error("Cant' get list of cubes: %s", resp.status_code)
                raise RuntimeError(f"Cant' get list of cubes: {resp.status_code=}")
                return None
            resp_j = json.loads(resp.content)
            return resp_j
