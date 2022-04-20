import numpy as np

from pyreal.explainers import TimeSeriesImportanceBase


class OcclusionImportance(TimeSeriesImportanceBase):
    """
    OcclusionImportance object.

    An OcclusionImportance object judges the relative importance or saliency of each timestep
    value by iteratively occluding windows of data, and calculating the resulting change in model
    prediction.

    Currently, only classification model explanation is supported.

    Args:
        model (string filepath or model object):
           Filepath to the pickled model to explain, or model object with .predict() function
        x_train_orig (DataFrame of size (n_instances, length of series)):
            Training set in original form.
        window_size (int):
            The size of the interval.
        shap_type (string, one of ["kernel", "linear"]):
            Type of shap algorithm to use. If None, SHAP will pick one.
        **kwargs: see base Explainer args
    """

    def __init__(self, model, x_train_orig,
                 window_size=1, **kwargs):
        self.window_size = window_size
        self.explainer_input_size = None
        super(OcclusionImportance, self).__init__(model, x_train_orig, **kwargs)

    def fit(self):
        """
        Fit the contribution explainer
        """
        return self

    def get_contributions_slow(self, x_orig, width=5, k="avg"):
        """
        Calculate the explanation of each feature in x using SHAP.

        Args:
            x_orig (DataFrame of shape (n_instances, n_features)):
               The input to be explained
        Returns:
            DataFrame of shape (n_instances, n_features):
                 The contribution of each feature
        """
        data_length = x_orig.shape[1]
        v = np.zeros((x_orig.shape[1], len(self.classes)))
        sig = np.copy(x_orig)

        for i in range(1, width):
            occ_test_signal = np.copy(sig)
            if(k == "avg"):
                occ_test_signal[:, 0:i] = np.average([occ_test_signal[:,0], occ_test_signal[:,i-1]])
            else:
                occ_test_signal[:, 0:i] = k
            pred = self.model.predict_proba(occ_test_signal).reshape(-1)
            for j in range(i, data_length):
                v[j] += pred

        for i in range(data_length - width):
            occ_test_signal = np.copy(sig)
            if (k == "avg"):
                occ_test_signal[:, i:i + width] = np.average([occ_test_signal[:, i], occ_test_signal[:, i+width-1]])
            else:
                occ_test_signal[:, i:i + width] = k
            pred = self.model.predict_proba(occ_test_signal).reshape(-1)
            for j in range(0, i):
                v[j] += pred
            for j in range(i+width, data_length):
                v[j] += pred

        for i in range(1, width):
            occ_test_signal = np.copy(sig)
            if (k == "avg"):
                occ_test_signal[:, data_length - i:data_length] = \
                    np.average([occ_test_signal[:, data_length-i], occ_test_signal[:, data_length-1]])
            else:
                occ_test_signal[:, data_length - i:data_length] = k
            pred = self.model.predict_proba(occ_test_signal).reshape(-1)
            for j in range(0, data_length - i):
                v[j] += pred

        print(v)

        predicted_class = self.model.predict(np.copy(sig))[0][0]
        class_index = self.classes.index(predicted_class)
        importance = v[:, class_index]
        max = np.amax(importance)
        min = np.amin(importance)
        importance = (importance - min) / (max - min)

        return importance.reshape(-1)

    def get_contributions(self, x_orig, width=5, k="avg"):
        """
        Calculate the explanation of each feature in x using SHAP.

        Args:
            x_orig (DataFrame of shape (n_instances, n_features)):
               The input to be explained
        Returns:
            DataFrame of shape (n_instances, n_features):
                 The contribution of each feature
        """
        data_length = x_orig.shape[1]
        v = np.zeros((x_orig.shape[1], len(self.classes)))
        sig = np.copy(x_orig)

        for i in range(1, width):
            #occ_test_signal = np.copy(sig)
            original_signal = sig[:, 0:i]
            if k == "avg":
                sig[:, 0:i] = np.average([sig[:, 0], sig[:, i-1]])
            else:
                sig[:, 0:i] = k
            pred = self.model.predict_proba(sig).reshape(-1)
            for j in range(i, data_length):
                v[j] += pred
            sig[:, 0:i] = original_signal

        for i in range(data_length - width):
            #occ_test_signal = np.copy(sig)
            original_signal = sig[:, 0:i]
            if (k == "avg"):
                sig[:, i:i + width] = np.average([sig[:, i], sig[:, i+width-1]])
            else:
                sig[:, i:i + width] = k
            pred = self.model.predict_proba(sig).reshape(-1)
            for j in range(0, i):
                v[j] += pred
            for j in range(i+width, data_length):
                v[j] += pred
            sig[:, 0:i] = original_signal

        for i in range(1, width):
            #occ_test_signal = np.copy(sig)
            original_signal = sig[:, 0:i]
            if (k == "avg"):
                sig[:, data_length - i:data_length] = \
                    np.average([sig[:, data_length-i], sig[:, data_length-1]])
            else:
                sig[:, data_length - i:data_length] = k
            pred = self.model.predict_proba(sig).reshape(-1)
            for j in range(0, data_length - i):
                v[j] += pred
            sig[:, 0:i] = original_signal

        predicted_class = self.model.predict(sig)[0][0]
        class_index = self.classes.index(predicted_class)
        importance = v[:, class_index]
        max = np.amax(importance)
        min = np.amin(importance)
        importance = (importance - min) / (max - min)

        return importance.reshape(-1)
