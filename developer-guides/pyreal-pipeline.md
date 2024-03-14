---
description: Understanding the Pyreal process
---

# Feature Space Pipeline

To generate interpretable explanationl, Pyreal follows a specific transformation pipeline:

1. Data is transformed from the **original feature space** to the **algorithm feature space**
   * x\_orig → `algorithm_transformers.transform()` → x\_algorithm
2. Data in the algorithm feature space is fed into the explanation algorithm, which results in an explanation in the algorithm feature space. The explanation algorithm may additionally transform data to the model-ready feature space to make predictions
   * x\_algorithm → `Explainer.produce()` → explanation\_algorithm
   * x\_algorithm → `model_transformers.transform()` → x\_model
3. The explanation is transformed to the original feature space by undoing the algorithm transforms
   * explanation\_algorithm → `algorithm_transformers.inverse_transform_explanation()` → explanation\_orig
4. The explanation is transformed to the interpretable feature space by running additional explanation transforms&#x20;
   * explanation\_orig → `interpret_transformers.transform_explanation()` → explanation\_interpret
5. For local explanations, the original data is transformed to the interpretable feature space
   * x\_orig → `interpret_transformers.transform()` → x\_interpret
