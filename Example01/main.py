from typing import Optional, Tuple
import logging
import numpy as np

from di_lib import di_app

logging.basicConfig(level=logging.DEBUG)
LOG = logging.getLogger(__name__)
# some comment
# some other comment
# 3d comment
class ExampleJob(di_app.DiAppSeismic3D):
    def __init__(self) -> None:
        super().__init__(in_name_par="Input Seismic3D Name", 
                out_name_par="New Seismic3D Name", out_names=["Example02 1", "Example02 2"])

    def compute(self, f_in_tup: Tuple[np.ndarray]) -> Tuple:
        f_in = f_in_tup[0]
        LOG.info(f"{self.cube_in} DT: {self.cube_in.time_step if self.cube_in else None}")
        return (f_in, f_in)

if __name__ == "__main__":
    LOG.debug(f"Starting job Example01")
    job = ExampleJob()
    res_final = job.run()
    LOG.info(f"{res_final}")
