import numpy as np

from pyreal.explainers.time_series import SaliencyBase


class OcclusionSaliency(SaliencyBase):
    """
    OcclusionSaliency object.

    An OcclusionSaliency object judges the relative importance or saliency of each timestep
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

    def __init__(self, model, x_train_orig, window_size=1, **kwargs):
        self.window_size = window_size
        self.explainer_input_size = None
        super(OcclusionSaliency, self).__init__(model, x_train_orig, **kwargs)

    def fit(self):
        """
        Fit the contribution explainer
        """
        return self

    def get_contributions(self, x_orig, width=5, k="avg"):
        """
        Calculate the explanation of each feature in x using SHAP.

        Args:
            x_orig (numpy array of shape (n_instances, n_features)):
               The input to be explained
        Returns:
            DataFrame of shape (n_instances, n_features):
                 The contribution of each feature
        """
        x_orig = self.transform_to_x_algorithm(x_orig)
        num_features = x_orig.shape[1]
        signal_length = x_orig.shape[2]

        sig = x_orig
        print(sig.shape)
        v = np.zeros((signal_length, num_features, len(self.model.predict(x_orig)[0])))

        for f in range(num_features):
            #sig = x_orig[f]
            for i in range(1, width):
                original_signal = sig[:, f, 0:i].copy()
                if k == "avg":
                    sig[:, f, 0:i] = np.average([sig[:, f, 0], sig[:, f, i - 1]])
                else:
                    sig[:, f, 0:i] = k
                pred = self.model.predict(sig).reshape(-1)
                for j in range(i, signal_length):
                    v[j, f] += pred
                sig[:, f, 0:i] = original_signal

            for i in range(signal_length - width):
                original_signal = sig[:, f, i:i + width].copy()
                if k == "avg":
                    sig[:, f, i:i + width] = np.average([sig[:, f, i], sig[:, f, i + width - 1]])
                else:
                    sig[:, f, i:i + width] = k
                pred = self.model.predict(sig).reshape(-1)
                for j in range(0, i):
                    v[j, f] += pred
                for j in range(i + width, signal_length):
                    v[j, f] += pred
                sig[:, f, i:i + width] = original_signal

            for i in range(1, width):
                original_signal = sig[:, f, signal_length - i:signal_length].copy()
                if k == "avg":
                    sig[:, f, signal_length - i: signal_length] = np.average(
                        [sig[:, f, signal_length - i], sig[:, f, signal_length - 1]]
                    )
                else:
                    sig[:, f, signal_length - i: signal_length] = k
                pred = self.model.predict(sig).reshape(-1)
                for j in range(0, signal_length - i):
                    v[j, f] += pred
                sig[:, f, signal_length - i: signal_length] = original_signal

        predicted_class = np.argmax(self.model.predict(x_orig))
        importance = v[:, :, predicted_class]
        max_val = np.amax(importance)
        min_val = np.amin(importance)
        importance = (importance - min_val) / (max_val - min_val)

        return importance.reshape(num_features, -1)
