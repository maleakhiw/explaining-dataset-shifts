#-------------------------------------------------------------------------------
# SHIFT STATISTICAL TEST
# 
# Author: Maleakhi A. Wijaya
# Description: This file contains various statistical tests functions.
# Credit: Some test codes were taken from the failing-loudly repository, written
#    by Stephan Rabanser.
#-------------------------------------------------------------------------------

from scipy.stats import ks_2samp, binom_test, chisquare, chi2_contingency, anderson_ksamp
from scipy.stats import chi2
from scipy.spatial import distance
import scipy.io
# from torch_two_sample import *
import numpy as np

from constants import *


#-------------------------------------------------------------------------------

def test_shift_bin(n_successes, n, p):
    """
    Binomial test for domain classifier. Used to check whether accuracy is statistically
    significant.

    :param n_successes: number of correctly predicted instances
    :param n: number of predictions made (test instances)
    :param p: hypothesised probability of success

    :return: pvalue
    """
    p_val = binom_test(n_successes, n, p)
    return p_val


def one_dimensional_test(X1, X2, test_type):
    """
    Given two matrices (each matrix is of size n x features), we conduct one 
    dimensional statistical test to each of the component using Kolmogorov-Smirnov
    or Anderson Darling to check whether each component comes from the same distribution.

    :param X1: matrix of data1 of size n x number of features.
    :param X2: matrix of data2 of size n x number of features.
    :param test_type: specify the one dimensional test to compare distributions
        of components (can be OneDimensionalTest.KS or OneDimensionalTest.AD)

    :return: minimum p values from all individual test, where we will check if
        its value less than alpha / number of components (Bonferroni correction).
    """
    p_vals = []
    t_vals = []
    ppfs = []

    # For each dimension we conduct a separate statistical test
    # Iterate over components
    for i in range(X1.shape[1]):
        feature_X1 = X1[:, i]
        feature_X2 = X2[:, i]

        t_val, p_val = None, None

        if test_type == OneDimensionalTest.KS:
            # Compute KS statistic and p-value
            t_val, p_val = ks_2samp(feature_X1, feature_X2)

            # Best on wikipedia, the formula for ppf ks_2samp is
            c_alpha = 1.358
            n = len(feature_X1)
            m = len(feature_X2)
            ppf = c_alpha * np.sqrt((n+m)/(n*m))
        else:
            t_val, critical_values, p_val = anderson_ksamp([feature_X1.tolist(), feature_X2.tolist()])
            ppf = critical_values[2]

        p_vals.append(p_val)
        t_vals.append(t_val)
        ppfs.append(ppf)

    # Apply the Bonferroni correction to bound the family-wise error rate. 
    # This can be done by picking the minimum p-value from all individual tests.
    p_vals = np.array(p_vals)
    p_val = np.min(p_vals)

    return p_val, p_vals, t_vals, ppfs


def test_chi2_shift(X1, X2, nb_classes):
    """
    Used for testing BBSD with hard threshold. Theoretically we conduct categorical
    chi2 test to test whether the distribution of the class follow theoretical
    chi2 distributions.

    :param X1: matrix of data1 of size n x number of features.
    :param X2: matrix of data2 of size n x number of features.
    :param nb_classes: number of classes (for degree of freedom).

    :return: p-value.
    """

    # Calculate observed and expected counts
    freq_exp = np.zeros(nb_classes)
    freq_obs = np.zeros(nb_classes)

    unique_X1, counts_X1 = np.unique(X1, return_counts=True)
    total_counts_X1 = np.sum(counts_X1)
    unique_X2, counts_X2 = np.unique(X2, return_counts=True)
    total_counts_X2 = np.sum(counts_X2)

    for i in range(len(unique_X1)):
        freq_exp[unique_X1[i]] = counts_X1[i]
        
    for i in range(0, len(unique_X2)):
        i = int(i)
        freq_obs[unique_X2[i]] = counts_X2[i]

    if np.amin(freq_exp) == 0 or np.amin(freq_obs) == 0:
        # The chi-squared test using contingency tables is not well defined if zero-element classes exist, which
        # might happen in the low-sample regime. In this case, we calculate the standard chi-squared test.
        for i in range(0, len(unique_X1)):
            i = int(i)
            val = counts_X1[i] / total_counts_X1 * total_counts_X2
            freq_exp[unique_X1[i]] = val
        chi, p_val = chisquare(freq_obs, f_exp=freq_exp)
        ppf = chi2.ppf(0.95, nb_classes-1)
    else:
        # In almost all cases, we resort to obtaining a p-value from the chi-squared test's contingency table.
        freq_conc = np.array([freq_exp, freq_obs])
        chi, p_val, dof, _ = chi2_contingency(freq_conc)
        ppf = chi2.ppf(0.95, dof)

    return chi, p_val, ppf


# Note: we omit this for now, too slow and performances are similar to one dimensional
#    test. Hence, not much benefits. We included in the results that we reported.
# def multi_dimensional_test(X1, X2):
#     """
#     Perform MMD multi dimensional test. See paper for more details.

#     :param X1: matrix of data1 of size n x number of features
#     :param X2: matrix of data2 of size n x number of features

#     :return: p-value
#     """

#     # torch_two_sample somehow wants the inputs to be explicitly casted to float 32.
#     X1 = X1.astype(np.float32)
#     X2 = X2.astype(np.float32)

#     p_val = None

#     # Do the MMD test
#     mmd_test = MMDStatistic(len(X1), len(X2))

#     # As per the original MMD paper, the median distance between all points in the aggregate sample from both
#     # distributions is a good heuristic for the kernel bandwidth, which is why compute this distance here.
#     if len(X1.shape) == 1:
#         X1 = X1.reshape((len(X1),1))
#         X2 = X2.reshape((len(X2),1))
#         all_dist = distance.cdist(X1, X2, 'euclidean')
#     else:
#         all_dist = distance.cdist(X1, X2, 'euclidean')
#     median_dist = np.median(all_dist)

#     # Calculate MMD.
#     t_val, matrix = mmd_test(torch.autograd.Variable(torch.tensor(X1)),
#                                 torch.autograd.Variable(torch.tensor(X2)),
#                                 alphas=[1/median_dist], ret_matrix=True)
#     p_val = mmd_test.pval(matrix)
        
#     return p_val