from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder

from pyreal.transformers import ColumnDropTransformer, OneHotEncoder, Transformer, fit_transformers


def sklearn_pipeline_to_pyreal_transformers(pipeline, X_train=None, verbose=0):
    """
    Convert a sklearn pipeline to a Pyreal pipeline
    Args:
        pipeline (sklearn.pipeline.Pipeline or list of sklearn transformers):
            The pipeline/transformers to convert
        X_train (pandas.DataFrame): The data to fit the transformers to. If None, transformers
            will not be fitted and may need to be manually fitted before use
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
        if isinstance(step, list):
            for substep in step:
                process_pipeline(substep, columns)
        elif isinstance(step, SklearnPipeline):
            for _, substep in step.steps:
                process_pipeline(substep, columns)
        elif isinstance(step, ColumnTransformer):
            if hasattr(step, "transformers_"):  # This ColumnTransformer has been fitted
                for _, substep, subcolumns in step.transformers_:
                    process_pipeline(substep, subcolumns)
            else:
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
    if X_train is not None:
        fit_transformers(pyreal_pipeline, X_train)
    return pyreal_pipeline
