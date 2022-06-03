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

    np2d_to_df
    np2d_to_df.fit
    np2d_to_df.data_transform
    pd2d_to_df
    pd2d_to_df.fit
    pd2d_to_df.data_transform
    np3d_to_df
    np3d_to_df.fit
    np3d_to_df.data_transform
    df_to_np3d
    df_to_np3d.data_transform
    df_to_np2d
    df_to_np2d.data_transform
    nested_to_df
    nested_to_df.fit
    nested_to_df.data_transform
    nested_to_np3d
    nested_to_np3d.data_transform
    df_to_nested
    df_to_nested.data_transform

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

