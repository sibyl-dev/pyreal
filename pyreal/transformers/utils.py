from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder
from sklearn.compose import ColumnTransformer

from pyreal.transformers import OneHotEncoder
from pyreal.transformers import Transformer
from sklearn.pipeline import Pipeline


def sklearn_pipeline_to_pyreal(pipeline, verbose=0):
    """
    Convert a sklearn pipeline to a Pyreal pipeline
    Args:
        pipeline (sklearn.pipeline.Pipeline): The pipeline to convert
        verbose (int): The verbosity level. 0 for no output, 1 for detailed output
    Returns:
        list of pyreal.transformers: The Pyreal pipeline
    """
    pyreal_pipeline = []
    for _, step in pipeline.steps:
        if not hasattr(step, "transform"):
            if verbose:
                print(f"Skipping step {step} as it does not appear to be a transformer")
            continue
        if isinstance(step, ColumnTransformer):
            for _, transformer, columns in step.transformers:
                if isinstance(transformer, SklearnOneHotEncoder):
                    pyreal_pipeline.append(OneHotEncoder(columns=columns))
                    if verbose:
                        print(f"Adding OneHotEncoder for columns {columns}")
                else:
                    pyreal_pipeline.append(
                        Transformer(wrapped_transformer=transformer, columns=columns)
                    )
                    if verbose:
                        print(f"Adding transformer {transformer} for columns {columns}")
        elif isinstance(step, SklearnOneHotEncoder):
            pyreal_pipeline.append(OneHotEncoder())
            if verbose:
                print(f"Adding OneHotEncoder")
        else:
            pyreal_pipeline.append(Transformer(wrapped_transformer=step))
            if verbose:
                print(f"Adding transformer {step}")
    return pyreal_pipeline
