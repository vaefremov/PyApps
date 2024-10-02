from typing import Optional, Tuple
import logging
import numpy as np

from di_lib import di_app
from di_lib.di_app import Context
from di_lib.seismic_cube import DISeismicCube
from di_lib.attribute import DIHorizon3D, DIAttribute2D

import time

logging.basicConfig(level=logging.DEBUG)
LOG = logging.getLogger(__name__)


def compute_attribute(cube_in: DISeismicCube, hor_in: DIHorizon3D) -> Optional[np.ndarray]:
    dt = hor_in.get_data()
    return dt

class ExampleHor1(di_app.DiAppSeismic3D):
    def __init__(self) -> None:
        super().__init__(in_name_par="Input Seismic3D Names", 
                out_name_par="New Name", out_names=[])

    def compute(self, f_in_tup: Tuple[np.ndarray], context: Context) -> Tuple:
        raise NotImplementedError("Shouldn't be called in this application!")

if __name__ == "__main__":
    LOG.debug(f"Starting job ExampleHor1")
    tm_start = time.time()
    job = ExampleHor1()

    cube_in = job.open_input_dataset()
    hor_name = job.description()["Horizon"]
    hor = job.session.get_horizon_3d(cube_in.geometry_name, hor_name)
    new_hor = job.session.create_horizon_3d_writer_as_other(hor, job.description()["New Name"])

    # Here some computations resulting in ar
    dt = compute_attribute(cube_in, hor)

    new_hor.write_data(dt)

    LOG.info(f"Processing time (s): {time.time() - tm_start}")
    