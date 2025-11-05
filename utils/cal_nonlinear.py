import numpy as np
import antropy as ant
from lempel_ziv_complexity import lempel_ziv_complexity


class Nonlinear_tool():

    def __init__(self, x) -> None:
        self.x = x
    
    def apen(self):
        return ant.app_entropy(self.x)
    
    def sample_en(self):
        return ant.sample_entropy(self.x)

    def lz_complexity(self):
        return lempel_ziv_complexity(str(self.x))
    