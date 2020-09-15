from sdv import SDV
from sdv import Metadata
import numpy as np
import pandas as pd
from ctgan import CTGANSynthesizer


def generate_data(X, cat_cols, N, return_result=False, save_file=None):
    ctgan = CTGANSynthesizer()
    ctgan.fit(X, cat_cols, epochs=10)
    samples = ctgan.sample(N)
    if return_result:
        return samples

