import logging
import numpy as np
import requests
import json
import struct

LOG = logging.getLogger(__name__)

SAMPLE_BYTE_LEN = 4 # Corresponds to lsb format of data
MAXFLOAT = 3.40282347e+38 ## stands for undefined values of parameters
MAXFLOAT09 = 0.9*3.40282347e+38 ## stands for undefined values of parameters


class DIPlanar3D:
    def __init__(self, project_id: int, geometry_name: str, name: str) -> None
        self.server_url = ""
        self.token = ""
        self.origin = None
        self.v_i = None
        self.v_x = None
        self.n_i = -1
        self.n_x = -1
        self.norm_v_i = None
        self.norm_v_x = None
        self.project_id = project_id
        self.geometry_name = geometry_name
        self.name = name
        self.geometry_id = None
        self.cube_id = None
        self.min_i = -1
        self.min_x = -1
        self.n_layers = 0
        