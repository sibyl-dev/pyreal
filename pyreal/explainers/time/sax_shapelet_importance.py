from shap.utils._legacy import (
    convert_to_model,
    convert_to_link,
    IdentityLink,
)

from shap.utils import safe_isinstance
from scipy.special import binom
import numpy as np
import pandas as pd
import scipy as sp
import logging
import copy
import itertools
import warnings
import gc

from tqdm.auto import tqdm
from shap import KernelExplainer as Kernel

import time

from pyreal.transformers import is_valid_dataframe, df_to_np2d, np2d_to_df


log = logging.getLogger("shap")


class SAXKernel(Kernel):
    def __init__(self, model, data, link=IdentityLink(), **kwargs):
        # convert incoming inputs to standardized iml objects
        self.time = time.time()
        self.link = convert_to_link(link)
        self.model = convert_to_model(model)
        self.keep_index = kwargs.get("keep_index", False)
        self.keep_index_ordered = kwargs.get("keep_index_ordered", False)
        # using pyreal time-series format data
        # only consider one variable for now
        if not is_valid_dataframe(data):
            raise ValueError("Data is not in valid format!")

        self.data = data
        model_null = self.model.f(self.data)

        # warn users about large background data sets
        if len(self.data.index) > 100:
            log.warning(
                "Using "
                + str(len(self.data.index))
                + " background data samples could cause "
                + "slower run times. Consider using shap.sample(data, K) or shap.kmeans(data,"
                " K) to "
                + "summarize the background as K samples."
            )

        # init our parameters
        self.N = self.data.shape[0]
        self.V, self.T = self.data.columns.levshape
        self.linkfv = np.vectorize(self.link.f)
        self.nsamplesAdded = 0
        self.nsamplesRun = 0

        # find E_x[f(x)]
        if isinstance(model_null, (pd.DataFrame, pd.Series)):
            model_null = np.squeeze(model_null.values)
        if safe_isinstance(model_null, "tensorflow.python.framework.ops.EagerTensor"):
            model_null = model_null.numpy()
        self.fnull = np.average(model_null, 0)
        self.expected_value = self.linkfv(self.fnull)

        # see if we have a vector output
        self.vector_out = True
        if len(self.fnull.shape) == 0:
            self.vector_out = False
            self.fnull = np.array([self.fnull])
            self.D = 1
            self.expected_value = float(self.expected_value)
        else:
            self.D = self.fnull.shape[0]

    def shap_values(self, X, groups, seqMean, seqStd, **kwargs):
        """Estimate the SHAP values for a set of samples."""
        # columns = X.columns.get_level_values(0).unique()
        # timestamps = X.columns.get_level_values(1).unique()
        # instance_idxs = X.index

        # single instance
        if len(X.index) == 1:

            explanation = self.explain(X, groups, seqMean, seqStd, **kwargs)
            return explanation

        # explain the whole dataset
        else:
            explanations = []
            for i in tqdm(range(X.shape[0]), disable=kwargs.get("silent", False)):
                data = X.iloc[i : i + 1]
                explanations.append(self.explain(data, groups, seqMean, seqStd, **kwargs))
                if kwargs.get("gc_collect", False):
                    gc.collect()

            return explanations

    def explain(self, instance, groups, seqMean, seqStd, **kwargs):
        """
        Explain the model's prediction on the input instance.

        Args:
            instance (pd.DataFrame):
                Instance to be explain, should have the same dimensionality as the original
                dataset except for the Index (number of instances)
            groups (list of ndarray or 2d ndarray):
                Each entry (an 1d ndarray) represents the range of the shapelet
        """
        # Create feature groups from word indices, (each span of the word count as a group)
        self.indexGroups = groups
        self.M = len(groups)

        # convert to numpy array as it is much faster if not jagged array (all groups of same length)
        if self.indexGroups and all(len(groups[i]) == len(groups[0]) for i in range(self.M)):
            self.indexGroups = np.array(self.indexGroups)
            # further performance optimization in case each group has a single value
            if self.indexGroups.shape[1] == 1:
                self.indexGroups = self.indexGroups.flatten()

        model_out = self.model.f(instance)

        if isinstance(model_out, (pd.DataFrame, pd.Series)):
            model_out = model_out.values
        self.fx = model_out[0]

        if not self.vector_out:
            self.fx = np.array([self.fx])

        # if no features vary then no feature has an effect
        if self.M == 0:
            phi = np.zeros((self.T, self.D))
            phi_var = np.zeros((self.T, self.D))

        # if only one feature varies then it has all the effect
        elif self.M == 1:
            phi = np.zeros((self.T, self.D))
            phi_var = np.zeros((self.T, self.D))
            diff = self.link.f(self.fx) - self.link.f(self.fnull)
            for d in range(self.D):
                phi[self.indexGroups[0], d] = diff[d]

        # if more than one feature varies then we have to do real work
        else:
            self.l1_reg = kwargs.get("l1_reg", "auto")

            # pick a reasonable number of samples if the user didn't specify how many they wanted
            self.nsamples = kwargs.get("nsamples", "auto")
            if self.nsamples == "auto":
                self.nsamples = 2 * self.M + 2**11

            # if we have enough samples to enumerate all subsets then ignore the unneeded samples
            self.max_samples = 2**30
            if self.M <= 30:
                self.max_samples = 2**self.M - 2
                if self.nsamples > self.max_samples:
                    self.nsamples = self.max_samples

            # reserve space for some of our computations
            self.allocate(instance)

            # weight the different subset sizes
            num_subset_sizes = np.int(np.ceil((self.M - 1) / 2.0))
            num_paired_subset_sizes = np.int(np.floor((self.M - 1) / 2.0))
            weight_vector = np.array(
                [(self.M - 1.0) / (i * (self.M - i)) for i in range(1, num_subset_sizes + 1)]
            )
            weight_vector[:num_paired_subset_sizes] *= 2
            weight_vector /= np.sum(weight_vector)
            log.debug("weight_vector = {0}".format(weight_vector))
            log.debug("num_subset_sizes = {0}".format(num_subset_sizes))
            log.debug("num_paired_subset_sizes = {0}".format(num_paired_subset_sizes))
            log.debug("M = {0}".format(self.M))

            # fill out all the subset sizes we can completely enumerate
            # given nsamples*remaining_weight_vector[subset_size]
            num_full_subsets = 0
            num_samples_left = self.nsamples
            group_inds = np.arange(self.M, dtype="int64")
            mask = np.zeros(self.M)
            remaining_weight_vector = copy.copy(weight_vector)
            for subset_size in range(1, num_subset_sizes + 1):

                # determine how many subsets (and their complements) are of the current size
                nsubsets = binom(self.M, subset_size)
                if subset_size <= num_paired_subset_sizes:
                    nsubsets *= 2
                log.debug("subset_size = {0}".format(subset_size))
                log.debug("nsubsets = {0}".format(nsubsets))
                log.debug(
                    "self.nsamples*weight_vector[subset_size-1] = {0}".format(
                        num_samples_left * remaining_weight_vector[subset_size - 1]
                    )
                )
                log.debug(
                    "self.nsamples*weight_vector[subset_size-1]/nsubsets = {0}".format(
                        num_samples_left * remaining_weight_vector[subset_size - 1] / nsubsets
                    )
                )

                # see if we have enough samples to enumerate all subsets of this size
                if (
                    num_samples_left * remaining_weight_vector[subset_size - 1] / nsubsets
                    >= 1.0 - 1e-8
                ):
                    num_full_subsets += 1
                    num_samples_left -= nsubsets

                    # rescale what's left of the remaining weight vector to sum to 1
                    if remaining_weight_vector[subset_size - 1] < 1.0:
                        remaining_weight_vector /= 1 - remaining_weight_vector[subset_size - 1]

                    # add all the samples of the current subset size
                    w = weight_vector[subset_size - 1] / binom(self.M, subset_size)
                    if subset_size <= num_paired_subset_sizes:
                        w /= 2.0
                    for inds in itertools.combinations(group_inds, subset_size):
                        mask[:] = 0.0
                        mask[np.array(inds, dtype="int64")] = 1.0
                        self.addsample(instance, mask, w, seqMean, seqStd)
                        if subset_size <= num_paired_subset_sizes:
                            mask[:] = np.abs(mask - 1)
                            self.addsample(instance, mask, w, seqMean, seqStd)
                else:
                    break
            log.info("num_full_subsets = {0}".format(num_full_subsets))
            print(time.time() - self.time)

            # add random samples from what is left of the subset space
            nfixed_samples = self.nsamplesAdded
            samples_left = self.nsamples - self.nsamplesAdded
            log.debug("samples_left = {0}".format(samples_left))
            if num_full_subsets != num_subset_sizes:
                remaining_weight_vector = copy.copy(weight_vector)
                remaining_weight_vector[
                    :num_paired_subset_sizes
                ] /= 2  # because we draw two samples each below
                remaining_weight_vector = remaining_weight_vector[num_full_subsets:]
                remaining_weight_vector /= np.sum(remaining_weight_vector)
                log.info("remaining_weight_vector = {0}".format(remaining_weight_vector))
                log.info("num_paired_subset_sizes = {0}".format(num_paired_subset_sizes))
                ind_set = np.random.choice(
                    len(remaining_weight_vector),
                    4 * samples_left,
                    p=remaining_weight_vector,
                )
                ind_set_pos = 0
                used_masks = {}
                while samples_left > 0 and ind_set_pos < len(ind_set):
                    mask.fill(0.0)
                    ind = ind_set[
                        ind_set_pos
                    ]  # we call np.random.choice once to save time and then just read it here
                    ind_set_pos += 1
                    subset_size = ind + num_full_subsets + 1
                    mask[np.random.permutation(self.M)[:subset_size]] = 1.0

                    # only add the sample if we have not seen it before, otherwise just
                    # increment a previous sample's weight
                    mask_tuple = tuple(mask)
                    new_sample = False
                    if mask_tuple not in used_masks:
                        new_sample = True
                        used_masks[mask_tuple] = self.nsamplesAdded
                        samples_left -= 1
                        self.addsample(instance, mask, 1.0, seqMean, seqStd)
                    else:
                        self.kernelWeights[used_masks[mask_tuple]] += 1.0

                    # add the compliment sample
                    if samples_left > 0 and subset_size <= num_paired_subset_sizes:
                        mask[:] = np.abs(mask - 1)

                        # only add the sample if we have not seen it before, otherwise just
                        # increment a previous sample's weight
                        if new_sample:
                            samples_left -= 1
                            self.addsample(instance, mask, 1.0, seqMean, seqStd)
                        else:
                            # we know the compliment sample is the next one after the original sample, so + 1
                            self.kernelWeights[used_masks[mask_tuple] + 1] += 1.0

                # normalize the kernel weights for the random samples to equal the weight left after
                # the fixed enumerated samples have been already counted
                weight_left = np.sum(weight_vector[num_full_subsets:])
                log.info("weight_left = {0}".format(weight_left))
                self.kernelWeights[nfixed_samples:] *= (
                    weight_left / self.kernelWeights[nfixed_samples:].sum()
                )
                print(time.time() - self.time)

            # execute the model on the synthetic samples we have created
            self.run()
            print(time.time() - self.time)

            # solve then expand the feature importance (Shapley value) vector to contain the non-varying features
            phi = np.zeros((self.T, self.D))
            phi_var = np.zeros((self.T, self.D))
            for d in range(self.D):
                vphi, vphi_var = self.solve(self.nsamples / self.max_samples, d)
                # TODO: accelerate feature -> timestep conversion with vectorization
                for i, group in enumerate(self.indexGroups):
                    phi[group, d] = vphi[i]
                    phi_var[group, d] = vphi_var[i]

        if not self.vector_out:
            phi = np.squeeze(phi, axis=1)
            phi_var = np.squeeze(phi_var, axis=1)

        return phi

    def allocate(self, instance):
        # TODO: modify the following to fit with Pyreal time-series data format

        data = df_to_np2d().fit_transform(instance)
        self.synth_data = np2d_to_df(
            var_name=instance.columns.get_level_values(0).unique(),
            timestamps=instance.columns.get_level_values(1).unique(),
        ).fit_transform(np.tile(data, (self.nsamples, 1)))

        self.maskMatrix = np.zeros((self.nsamples, self.M))
        self.kernelWeights = np.zeros(self.nsamples)
        self.y = np.zeros((self.nsamples, self.D))
        self.ey = np.zeros((self.nsamples, self.D))
        self.lastMask = np.zeros(self.nsamples)
        self.nsamplesAdded = 0
        self.nsamplesRun = 0

    def addsample(self, x, m, w, seqMean, seqStd):
        """
        Append one sample to the synthetic data buffer
        """
        offset = self.nsamplesAdded
        # TODO: use a general framework
        self.synth_data.iloc[offset] = df_to_np2d().fit_transform(x)
        if isinstance(self.indexGroups, (list,)):
            # each j corresponds to a shapelet
            for j in range(self.M):
                # iterate over each timestep of the shapelet
                if m[j] == 1:
                    startPoint = self.indexGroups[j][0]
                    # TODO: speed this up with the fact that the timesteps here are contiguous
                    for k in self.indexGroups[j]:
                        self.synth_data.loc[offset, (slice(None), k)] = np.random.normal(
                            np.squeeze(seqMean[..., startPoint]),
                            np.squeeze(seqStd[..., startPoint]),
                        )

        else:
            # for non-jagged numpy array we can significantly boost performance
            mask = m.astype(bool)
            groups = self.indexGroups[mask]
            if len(groups.shape) == 2:
                for group in groups:
                    startPoint = group[0]
                    self.synth_data.loc[offset, (slice(None), group)] = np.random.normal(
                        np.squeeze(seqMean[..., startPoint]),
                        np.squeeze(seqStd[..., startPoint]),
                        len(group),
                    )
        self.maskMatrix[self.nsamplesAdded, :] = m
        self.kernelWeights[self.nsamplesAdded] = w
        self.nsamplesAdded += 1

    def run(self):
        num_to_run = self.nsamplesAdded - self.nsamplesRun
        data = self.synth_data.iloc[self.nsamplesRun : self.nsamplesAdded]
        if self.keep_index:
            index = self.synth_data_index[self.nsamplesRun : self.nsamplesAdded]
            index = pd.DataFrame(index, columns=[self.data.index_name])
            data = pd.DataFrame(data, columns=self.data.group_names)
            data = pd.concat([index, data], axis=1).set_index(self.data.index_name)
            if self.keep_index_ordered:
                data = data.sort_index()
        modelOut = self.model.f(data)
        if isinstance(modelOut, (pd.DataFrame, pd.Series)):
            modelOut = modelOut.values
        self.y[self.nsamplesRun : self.nsamplesAdded, :] = np.reshape(
            modelOut, (num_to_run, self.D)
        )
        self.ey = self.y
