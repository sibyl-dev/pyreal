from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder
from sklearn.compose import ColumnTransformer

from pyreal.transformers import OneHotEncoder, ColumnDropTransformer
from pyreal.transformers import Transformer
from sklearn.pipeline import Pipeline as SklearnPipeline


def sklearn_pipeline_to_pyreal(pipeline, verbose=0):
    """
    Convert a sklearn pipeline to a Pyreal pipeline
    Args:
        pipeline (sklearn.pipeline.Pipeline): The pipeline to convert
        verbose (int): The verbosity level. 0 for no output, 1 for detailed output
    Returns:
        list of pyreal.transformers: The Pyreal pipeline
    """

    def log(transformer, columns):
        if not verbose:
            return
        if columns is not None:
            print(f"Adding {transformer} for columns {columns}")
        else:
            print(f"Adding {transformer}")

    pyreal_pipeline = []

    def process_pipeline(step, columns=None):
        if isinstance(step, SklearnPipeline):
            for _, substep in step.steps:
                process_pipeline(substep, columns)
        elif isinstance(step, ColumnTransformer):
            for _, substep, subcolumns in step.transformers:
                process_pipeline(substep, subcolumns)
        elif step == "drop":
            pyreal_pipeline.append(ColumnDropTransformer(columns=columns))
            log("ColumnDropTransformer", columns)
        elif isinstance(step, SklearnOneHotEncoder):
            pyreal_pipeline.append(OneHotEncoder(columns=columns))
            log("OneHotEncoder", columns)
        elif not hasattr(step, "transform"):
            if verbose:
                print(f"Skipping step {step} as it does not appear to be a transformer")
        else:
            pyreal_pipeline.append(Transformer(wrapped_transformer=step, columns=columns))
            log(step, columns)

    process_pipeline(pipeline)
    return pyreal_pipeline
