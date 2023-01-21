import numpy as np
import pandas as pd
from _sax_kernel import SAXKernel

from pyreal.explainers.time_series import SaliencyBase
from pyreal.transformers import SAXTransformer, is_valid_dataframe
from pyreal.types.explanations.feature_based import FeatureContributionExplanation


class UnivariateSAXShaplet(SaliencyBase):
    """
    UnivariateSAXShaplet object.
    """

    def __init__(self, model, x_train_orig, n_bins=4, width=24, word_length=4, **kwargs):
        """

        Args:
            model (string filepath or model object):
                Filepath to the pickled model to explain, or model object with .predict() function
            x_train_orig (DataFrame of size (n_instances, length of series)):
                # TODO: does this fit the Transformer pipeline ??
                Training set in original form.
            n_bins (int, default = 4):
                The number of bins to produce. It must be between 2 and
                ``min(n_timestamps, 26)``.
            width (int)
                Length of each SAX word window
            word_length (int):
                The number of characters in each representation of word, which
                is reduced from each window.
        """
        self.sax_transformer = SAXTransformer(n_bins, width, word_length)
        self.explainer = None
        super(UnivariateSAXShaplet, self).__init__(model, x_train_orig, **kwargs)

    def fit(self, word_thresh=0.3):
        """
        Fit the SAX transformer and the subsequent SHAP kernel.
        """
        # The following generates the SAX representations of every signal in `x_train_orig`.
        # Note that each data instance would contain multiple SAX words.
        sax_list = self.sax_transformer.fit_transform(self._x_train_orig)
        # The shape of `sax_list` should be (n_instances, n_variables=1, n_words, word_length)
        # where n_words is the number of words in a single signal
        n_instances, n_variables, n_sequences, word_length = sax_list.shape
        # TODO: If buggy, uncomment the following line
        # sax_list = sax_list.tolist()
        # sax_dict counts the occurrence of each word in the dataset
        sax_dict = {}
        for i in range(n_instances):
            for t in range(n_sequences):
                word = "".join(str(x) for x in sax_list[i][0][t])
                if word not in sax_dict.keys():
                    sax_dict[word] = 1
                else:
                    sax_dict[word] += 1
        # now we want to filter the words that appear more than `word_thresh*n_instances`.
        self.sax_set = set()  # This is the final words list
        for word, count in sax_dict.items():
            if count >= word_thresh * n_instances:
                self.sax_set.add(word)

        dataset = self.transform_to_x_model(self._x_train_orig)
        self.explainer = SAXKernel(self.model, dataset)

        return self

    def get_contributions(self, x_orig):
        """
        Calculate the explanation of each feature in x using occlusion.
        Args:
            x_orig (DataFrame of shape (1, n_features)):
               The input to be explained
        Returns:
            DataFrame of shape (n_classes, n_features):
                 The contribution of each feature to each class prediction.
        """
        x_algo = self.transform_to_x_algorithm(x_orig)
        if isinstance(x_algo, pd.Series):
            x_algo = x_algo.to_frame().T

        if x_algo.shape[0] > 1:
            raise ValueError(
                "UnivariateOcclusionSaliency.produce() can only take one row of input"
            )
        data_length = x_algo.shape[1]

        v = np.zeros((data_length, self.num_classes))
        sig = np.copy(x_algo)
        pred_orig = self.model.predict(self.transform_x_from_algorithm_to_model(sig)).reshape(-1)

        # Occlude the beginning of the sequence with smaller windows
        for i in range(1, self.width):
            pred = self._occlude_once(sig, 0, i, self.k)
            for j in range(0, i):
                v[j] += pred - pred_orig

        # Occlude the main body of the sequence with width-length windows
        for i in range(data_length - self.width + 1):
            pred = self._occlude_once(sig, i, i + self.width, self.k)
            for j in range(i, i + self.width):
                v[j] += pred - pred_orig

        # Occlude the end of the sequence with smaller windows
        for i in range(1, self.width):
            pred = self._occlude_once(sig, data_length - i, data_length, self.k)
            for j in range(data_length - i, data_length):
                v[j] += pred - pred_orig

        importance = v / self.width

        if isinstance(x_algo, pd.DataFrame):
            importances_df = pd.DataFrame(importance, index=x_algo.columns).T
        else:
            importances_df = pd.DataFrame(importance).T

        return FeatureContributionExplanation(importances_df)

    def _occlude_once(self, sig, win_min, win_max, k):
        """
        Occlude one window and return the resulting model prediction
        """
        occ_test_signal = np.copy(sig)
        if self.k == "avg":
            occ_test_signal[:, win_min:win_max] = np.average(
                [occ_test_signal[:, win_min], occ_test_signal[:, win_max - 1]]
            )
        elif self.k == "remove":
            mask = np.ones_like(occ_test_signal, dtype=bool)
            mask[:, win_min:win_max] = False
            new_shape = list(sig.shape)
            new_shape[1] = -1
            occ_test_signal = occ_test_signal[mask, ...].reshape(new_shape)
        else:
            occ_test_signal[:, win_min:win_max] = k

        pred = self.model.predict(
            self.transform_x_from_algorithm_to_model(occ_test_signal)
        ).reshape(-1)
        return pred
