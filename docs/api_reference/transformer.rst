.. _pyreal.transformer:

Transformer
==============
.. currentmodule:: pyreal.transformers

Base Transformer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: api/

    Transformer
    Transformer.fit
    Transformer.data_transform
    Transformer.transform
    Transformer.fit_transform
    Transformer.transform_explanation

Feature Select Transformer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: api/

    FeatureSelectTransformer
    FeatureSelectTransformer.fit
    FeatureSelectTransformer.data_transform
    FeatureSelectTransformer.transform
    FeatureSelectTransformer.fit_transform
    FeatureSelectTransformer.transform_explanation

Imputers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: api/

    MultiTypeImputer
    MultiTypeImputer.fit
    MultiTypeImputer.data_transform
    MultiTypeImputer.transform
    MultiTypeImputer.fit_transform
    MultiTypeImputer.transform_explanation

One-Hot Encoders
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: api/

    OneHotEncoder
    OneHotEncoder.fit
    OneHotEncoder.data_transform
    OneHotEncoder.transform
    OneHotEncoder.fit_transform
    OneHotEncoder.transform_explanation
    MappingsOneHotEncoder
    MappingsOneHotEncoder.fit
    MappingsOneHotEncoder.data_transform
    MappingsOneHotEncoder.transform
    MappingsOneHotEncoder.fit_transform
    MappingsOneHotEncoder.transform_explanation
    MappingsOneHotDecoder
    MappingsOneHotDecoder.fit
    MappingsOneHotDecoder.data_transform
    MappingsOneHotDecoder.transform
    MappingsOneHotDecoder.fit_transform
    MappingsOneHotDecoder.transform_explanation
    Mappings
    Mappings.generate_mappings

Time-Series Formatters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: api/

    MultiIndexFrameToNestedFrame
    MultiIndexFrameToNestedFrame.data_transform
    MultiIndexFrameToNumpy2d
    MultiIndexFrameToNumpy2d.data_transform
    MultiIndexFrameToNumpy3d
    MultiIndexFrameToNumpy3d.data_transform
    NestedFrameToMultiIndexFrame
    NestedFrameToMultiIndexFrame.fit
    NestedFrameToMultiIndexFrame.data_transform
    NestedFrameToNumpy3d
    NestedFrameToNumpy3d.data_transform
    Numpy2dToMultiIndexFrame
    Numpy2dToMultiIndexFrame.fit
    Numpy2dToMultiIndexFrame.data_transform
    Numpy2dToNestedFrame
    Numpy2dToNestedFrame.fit
    Numpy2dToNestedFrame.data_transform
    Numpy3dToMultiIndexFrame
    Numpy3dToMultiIndexFrame.fit
    Numpy3dToMultiIndexFrame.data_transform
    Numpy3dToNestedFrame
    Numpy3dToNestedFrame.data_transform
    Pandas2dToMultiIndexFrame
    Pandas2dToMultiIndexFrame.fit
    Pandas2dToMultiIndexFrame.data_transform

Type Casters
~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: api/

    BoolToIntCaster
    BoolToIntCaster.fit
    BoolToIntCaster.data_transform
    BoolToIntCaster.transform
    BoolToIntCaster.fit_transform
    BoolToIntCaster.transform_explanation

Wrappers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: api/

    DataFrameWrapper
    DataFrameWrapper.fit
    DataFrameWrapper.data_transform
    DataFrameWrapper.transform
    DataFrameWrapper.fit_transform
    DataFrameWrapper.transform_explanation

Time Series Padders
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: api/

    TimeSeriesPadder
    TimeSeriesPadder.fit
    TimeSeriesPadder.data_transform
    TimeSeriesPadder.transform
    TimeSeriesPadder.fit_transform
    TimeSeriesPadder.transform_explanation

