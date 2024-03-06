.. _pyreal.transformer:

Transformer
==============
.. currentmodule:: pyreal.transformers
.. toctree::
   :maxdepth: 1
   :glob:

    Transformer
    FeatureSelectTransformer
    ColumnDropTransformer
    MinMaxScaler
    StandardScaler
    Normalizer
    MultiTypeImputer
    OneHotEncoder
    MappingsOneHotEncoder
    MappingsOneHotDecoder
    Mappings
    MultiIndexFrameToNestedFrame
    MultiIndexFrameToNumpy2d
    MultiIndexFrameToNumpy3d
    NestedFrameToMultiIndexFrame
    NestedFrameToNumpy3d
    Numpy2dToMultiIndexFrame
    Numpy2dToNestedFrame
    Numpy3dToMultiIndexFrame
    Numpy3dToNestedFrame
    Pandas2dToMultiIndexFrame
    BoolToIntCaster
    DataFrameWrapper
    TimeSeriesPadder
    LatLongToPlace


Transformer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: Transformer
    :members: fit, data_transform, transform, fit_transform, inverse_transform, transform_explanation, inverse_transform_explanation

Feature Select Transformer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: FeatureSelectTransformer
    :members: fit, data_transform, fit_transform
.. autoclass:: ColumnDropTransformer
    :members: fit, data_transform, fit_transform

Scalers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: MinMaxScaler
    :members: fit, data_transform, fit_transform
.. autoclass:: StandardScaler
    :members: fit, data_transform, fit_transform
.. autoclass:: Normalizer
    :members: fit, data_transform, fit_transform

Imputers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: MultiTypeImputer
    :members: fit, data_transform, fit_transform

One-Hot Encoders
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: OneHotEncoder
    :members: fit, data_transform, fit_transform
.. autoclass:: MappingsOneHotEncoder
    :members: fit, data_transform, fit_transform
.. autoclass:: MappingsOneHotDecoder
    :members: fit, data_transform, fit_transform
.. autoclass:: Mappings
    :members: generate_mappings

Time-Series Formatters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: MultiIndexFrameToNestedFrame
    :members: fit, data_transform, fit_transform
.. autoclass:: MultiIndexFrameToNumpy2d
    :members: fit, data_transform, fit_transform
.. autoclass:: MultiIndexFrameToNumpy3d
    :members: fit, data_transform, fit_transform
.. autoclass:: NestedFrameToMultiIndexFrame
    :members: fit, data_transform, fit_transform
.. autoclass:: NestedFrameToNumpy3d
    :members: fit, data_transform, fit_transform
.. autoclass:: Numpy2dToMultiIndexFrame
    :members: fit, data_transform, fit_transform
.. autoclass:: Numpy2dToNestedFrame
    :members: fit, data_transform, fit_transform
.. autoclass:: Numpy3dToMultiIndexFrame
    :members: fit, data_transform, fit_transform
.. autoclass:: Numpy3dToNestedFrame
    :members: fit, data_transform, fit_transform
.. autoclass:: Pandas2dToMultiIndexFrame
    :members: fit, data_transform, fit_transform

Type Casters
~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: BoolToIntCaster
    :members: fit, data_transform, fit_transform


Wrappers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: DataFrameWrapper
    :members: fit, data_transform, fit_transform

Time Series Padders
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: TimeSeriesPadder
    :members: fit, data_transform, fit_transform

Geo Transformers
~~~~~~~~~~~~~~~~~~~~
.. autoclass:: LatLongToPlace
    :members: fit, data_transform, fit_transform

