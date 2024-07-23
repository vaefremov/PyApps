from typing import Optional, Tuple
import logging
import numpy as np

import numexpr as ne

from di_lib import di_app
from di_lib.di_app import Context

logging.basicConfig(level=logging.DEBUG)
LOG = logging.getLogger(__name__)
MAXFLOAT = float(np.finfo(np.float32).max)

class SeismicCalculation(di_app.DiAppSeismic3DMultiple):
    def __init__(self) -> None:
        super().__init__(in_name_par="seismic_3d",
                out_name_par="result_name", out_names=["formula1"])
        
        # Input datasets names are converted to the agreed upon format 
        # (the CR character in  "geometry\nname\nname2" replaced by "/"", geometry name omitted)
        cube_names_for_formula = ["/".join(nn.split("\n")[1:]) for nn in self.description[self.in_name_par]]
        self.formula = self.description["formula"]
        # Converting formula to the format that can be used in the compute() method context
        # Datasets names replaced with references to the corresponding fragment in f_in_tup
        # argument of compute(). Replacement are applied in reverse order of name lengths, most
        # long names replaced first.
        for num, nm in reversed(sorted(enumerate(cube_names_for_formula), key=lambda x: len(x[1]))):
            self.formula = self.formula.replace(nm, f"fintup{num}")
        self.formula = self.formula.lower()
        self.formula = self.formula.strip()
       
        LOG.info(f"\n ***FORMULA*** \nOriginal formula: {self.description['formula']} \nFinal formula: {self.formula}")
       
    def compute(self, f_in_tup: Tuple[np.ndarray], context: Context) -> Tuple:
        LOG.info(f"Computing {[f_in.shape for f_in in f_in_tup]}")
        if (f_in_tup[0]>= 0.1*MAXFLOAT).all() or (f_in_tup[0] == np.inf).all():
            LOG.info("***EMPTY***")
            np.nan_to_num(f_in_tup[0], inf=MAXFLOAT, copy=False)
            return (f_in_tup[0],)
        
        else:
            fintup = list(f_in_tup)
            for i, v in enumerate(f_in_tup):
                globals() ["fintup{}".format(i)] = np.where((fintup[i] >= 0.1*MAXFLOAT) | (fintup[i] == np.inf), np.nan, fintup[i])
        
            f_out = ne.evaluate(self.formula)
            np.nan_to_num(f_out, nan=MAXFLOAT, copy=False)
            return (f_out,)

if __name__ == "__main__":
    LOG.debug(f"Starting job")
    job = SeismicCalculation()
    res_final = job.run()
    LOG.info(f"{res_final}")
