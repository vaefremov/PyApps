from typing import Optional, Tuple
import logging
import numpy as np

from di_lib import di_app

logging.basicConfig(level=logging.DEBUG)
LOG = logging.getLogger(__name__)

class SeismicCalculation(di_app.DiAppSeismic3DMultiple):
    def __init__(self) -> None:
        super().__init__(in_name_par="seismic_3d",
                out_name_par="result_name", out_names=["formula"])
        
        # Input datasets names are converted to the agreed upon format 
        # (the CR character in  "geometry\nname\nname2" replaced by "/"", geometry name omitted)
        cube_names_for_formula = ["/".join(nn.split("\n")[1:]) for nn in self.description[self.in_name_par]]
        self.formula = self.description["formula"]
        # Converting formula to the format that can be used in the compute() method context
        # Datasets names replaced with references to the corresponding fragment in f_in_tup
        # argument of compute(). Replacement are applied in reverse order of name lengths, most
        # long names replaced first.
        for num, nm in reversed(sorted(enumerate(cube_names_for_formula), key=lambda x: len(x[1]))):
            self.formula = self.formula.replace(nm, f"f_in_tup[{num}]")

        LOG.info(f"Original formula: {self.description['formula']} Final formula: {self.formula}")

    def compute(self, f_in_tup: Tuple[np.ndarray]) -> Tuple:
        LOG.info(f"Computing {[f_in.shape for f_in in f_in_tup]}")
        f_out = eval(self.formula)
        return (f_out,)

if __name__ == "__main__":
    LOG.debug(f"Starting job")
    job = SeismicCalculation()
    res_final = job.run()
    LOG.info(f"{res_final}")
