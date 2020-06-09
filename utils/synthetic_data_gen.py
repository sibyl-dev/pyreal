from sdv import SDV
from sdv import Metadata


def generate_data(X, N, return_result=False, save_file=None):
    sdv = SDV()
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



