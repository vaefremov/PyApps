from typing import Optional, Tuple
import logging
import numpy as np

from di_lib import di_app
from di_lib.di_app import Context

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)

# added comment to start deploy
# added comment to start deploy

class Example03(di_app.DiAppSeismic3D2D):
    def __init__(self) -> None:
        super().__init__(in_name_par="seismic_3d", 
                in_line_geometries_par="Seismic2DGeometries", in_line_names_par="seismic_2d",
                out_name_par="result_name", out_names=["Example03 1", "Example03 2"])

    def compute(self, f_in_tup: Tuple[np.ndarray], context: Context) -> Tuple:
        f_in = f_in_tup[0]
        LOG.info(f"Computing {f_in.shape}")
        if len(f_in.shape) == 3: 
            # Cubes
            LOG.info(f"In cube info: {context.in_cube_params}")
            LOG.info(f"Input/output chunk: {context.in_cube_params['chunk']}")
        return (f_in, f_in)

if __name__ == "__main__":
    LOG.debug(f"Starting job")
    job = Example03()
    res_final = job.run()
    LOG.info(f"{res_final}")
