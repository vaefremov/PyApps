from typing import Optional, Tuple
import logging
import numpy as np

from di_lib import di_app

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)

class ExampleNewZStep(di_app.DiAppSeismic3DMultiple):
    def __init__(self) -> None:
        super().__init__(in_name_par="seismic_3d",
                out_name_par="result_name", out_names=["Example04-1"])
        self.z_step_out = self.description["z_step"]

    def compute(self, f_in_tup: Tuple[np.ndarray]) -> Tuple:
        # LOG.info(f"{self.cube_in} DT: {self.cube_in.time_step if self.cube_in else None}")
        f_in = f_in_tup[0]
        LOG.info(f"Computing {f_in.shape}")
        new_nz = self.output_cubes_parameters["nz"]
        new_shape = (f_in.shape[0], f_in.shape[1], new_nz)
        f_out = np.zeros(new_shape, dtype=f_in.dtype)
        return (f_out,)

if __name__ == "__main__":
    LOG.debug(f"Starting job")
    job = ExampleNewZStep()
    res_final = job.run()
    LOG.info(f"{res_final}")
