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
    '''sdv = SDV()
    tables = {'X': X}
    metadata = Metadata()
    metadata.add_table('X', data=tables['X'])
    sdv.fit(metadata, tables)

    samples = sdv.sample_all(N)
    results = samples['X']

    if save_file is not None:
        results.to_csv(save_file)
    if return_result:
        return results
    X_np = np.asanyarray(X, dtype=np.object)
    X_np_swap = X_np.swapaxes(0, -1)
    X_shape = X_np.shape[0]
    idx = np.random.choice(X_shape, X_shape, replace=False)
    X_np_swap = X_np_swap[...,idx]
    shuffled = X_np_swap.swapaxes(0, -1)

    #np.random.shuffle(np.transpose(X_np))
    shuffled = pd.DataFrame(shuffled[0:N], columns=X.columns)
    print(shuffled.head())
    if save_file is not None:
        shuffled.to_csv(save_file)
    if return_result:
        return shuffled'''


