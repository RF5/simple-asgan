import logging
import math
import random

import numpy as np
import scipy.stats
import torch
import torch.nn.functional as F
import torchaudio
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.metrics import mutual_info_score
from torch import Tensor

def inception_score(logprobs_gen: Tensor) -> Tensor:
    """
    Calculate inception score from `logprobs_gen` (bs, n_classes) of 
    log probabilities.

    Adapted from https://github.com/HazyResearch/state-spaces/
    """
    # Set seed
    state = np.random.RandomState(0)
    inds = torch.from_numpy(state.permutation(len(logprobs_gen)))

    # Shuffle probs_gen
    logprobs_gen = logprobs_gen[inds]

    # Split probs_gen into two halves
    probs_gen_1 = logprobs_gen[:len(logprobs_gen)//2]
    probs_gen_2 = logprobs_gen[len(logprobs_gen)//2:]

    # Calculate average label distribution for split 2
    N = len(probs_gen_1)
    N2 = len(probs_gen_2)
    b = probs_gen_2.logsumexp(dim=0, keepdim=True) + math.log(1/N2)

    # Compute the mean kl-divergence between the probability distributions
    # of the generated and average label distributions
    kl = F.kl_div(b.repeat(N, 1), probs_gen_1,
                     reduction='batchmean', log_target=True)

    # Compute the expected score
    is_score = kl.exp()
    return is_score


def _two_proportions_z_test(p1, n1, p2, n2, significance_level, z_threshold=None):
    # Taken from https://github.com/eitanrich/gans-n-gmms/blob/master/utils/ndb.py
    # Per http://stattrek.com/hypothesis-test/difference-in-proportions.aspx
    # See also http://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/binotest.htm
    p = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = np.sqrt(p * (1 - p) * (1/n1 + 1/n2))
    z = (p1 - p2) / se
    # Allow defining a threshold in terms as Z (difference relative to the SE) rather than in p-values.
    if z_threshold is not None:
        return abs(z) > z_threshold
    p_values = 2.0 * scipy.stats.norm.cdf(-1.0 * np.abs(z))    # Two-tailed test
    return p_values < significance_level


def ndb_score(feat_data: Tensor, feat_gen: Tensor, K=50, proportion=True, precomputed_kmeans=None):
    """ 
    Calculates the number of statistically distinct bins (NDB)
    from features of real data `feat_data` (N1, dim) and features of
    generated data `feat_gen` (N2, dim)

    Adapted from https://github.com/HazyResearch/state-spaces/blob/main/sashimi/sc09_classifier/test_speech_commands.py
    """
    if precomputed_kmeans is None:
        # Run K-Means cluster on feat_data with K=50
        kmeans = KMeans(n_clusters=K, random_state=0).fit(feat_data)
    else: kmeans = precomputed_kmeans

    # Get cluster labels for feat_data and feat_gen
    labels_data = kmeans.predict(feat_data)
    labels_gen = kmeans.predict(feat_gen)

    # Calculate number of data points in each cluster using np.unique
    counts_data = np.unique(labels_data, return_counts=True)[1]
    counts_gen = np.zeros_like(counts_data)
    values, counts = np.unique(labels_gen, return_counts=True)
    counts_gen[values] = counts

    # Calculate proportion of data points in each cluster
    prop_data = counts_data / len(labels_data)
    prop_gen = counts_gen / len(labels_gen)

    # Calculate number of bins with statistically different proportions
    different_bins = _two_proportions_z_test(prop_data, len(labels_data), prop_gen, len(labels_gen), 0.05)
    ndb = np.count_nonzero(different_bins)
    if proportion: ndb = ndb / K
    return ndb


def modified_inception_score2(probs_gen, n=10000):
    """
    Calculate modified inception score from `probs_gen` (bs, n_classes) of 
    probabilities.

    Adapted from https://github.com/HazyResearch/state-spaces/
    """
    # Set seed
    rn = np.random.RandomState(123)

    n_samples = len(probs_gen)

    all_kls = []
    for i in range(n):
        # Sample two prob vectors
        indices = rn.choice(np.arange(n_samples), size=2, replace=True)
        probs_gen_1 = probs_gen[indices[0]]
        probs_gen_2 = probs_gen[indices[1]]

        # Calculate their KL
        kl = scipy.stats.entropy(probs_gen_1, probs_gen_2)

        all_kls.append(kl)

    # Compute the score
    mis_score = np.exp(np.mean(all_kls))

    return mis_score

def fid(c_f, c_train):
    """ Compute FID for features of generated utterances `c_f` (N, dim) 
    and features from training set `c_train` (N, dim).
    """
    mu_real = c_train.mean(dim=0)
    cov_real = c_train.T.cov()
    mu_fake = c_f.mean(dim=0)
    cov_fake = c_f.T.double().cov().float()
    fid = calculate_frechet_distance(mu_fake.numpy(), cov_fake.numpy(), mu_real.numpy(), cov_real.numpy())
    return fid

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.

    Retrieved and adapted from 
    https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py
    """
    from scipy.linalg import sqrtm
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = sqrtm(sigma1.astype(np.float64).dot(sigma2.astype(np.float64)), disp=False)
    # covmean = covmean.astype(np.float32)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            logging.warning(f"Imaginary component {m} in FAD calculation. Returning NaN.")
            return float('nan')
            #raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)

def am_score(gen_probs: Tensor, train_probs: Tensor) -> float:
    """ Calculate AM Score from `gen_probs` (N1, n_classes) and `train_probs` (N2, n_classes) """

    gen_entropy = -(gen_probs*gen_probs.log()).sum(dim=-1) # (N,)
    expected_gen_entropy = gen_entropy.mean()

    mean_train = train_probs.double().mean(dim=0) # (n_classes, ), E[p(x)]
    mean_gen = gen_probs.double().mean(dim=0)  # (n_classes, ), E[p(G(z))]
    kl = F.kl_div(mean_gen.log(), mean_train.log(), # swapped as torch has them opposite way around
                  reduction='sum', log_target=True) # sum reduce since no expecation
    score = kl + expected_gen_entropy

    return score


def linear_separability(logits: Tensor, latents: Tensor, svm_random_state: int, linear_sep_trim_frac: float) -> Tensor:
    """ Calculate linear separability with SVM, StyleGAN1-style, given deep classifier class `logits` (N, n_classes) and
    all the latent vectors `latents` (N, dim). Additionally specify the fraction of most confident logits to keep for eval.
    """
    probs = F.softmax(logits.double(), dim=-1)
    confidences = probs.max(dim=-1).values
    inds = torch.topk(confidences, k= math.ceil(logits.shape[0]*linear_sep_trim_frac)).indices

    probs = probs[inds]
    latents = latents[inds]

    # Get linear classifier probabilities
    lin_clas = SVC(random_state=svm_random_state, probability=True, kernel='linear')
    lin_clas.fit(latents.numpy(), y=probs.argmax(dim=-1).numpy())
    lin_clas_probs = torch.from_numpy(lin_clas.predict_proba(latents.numpy()))

    # conditional_entropy  H(classifier_probs | svm_probs)
    # X = svm_probs
    # Y = clasifier probs
    
    mutual_info = mutual_info_score(lin_clas_probs.argmax(dim=-1).numpy(), probs.argmax(dim=-1).numpy()) # In nats
    p = probs.mean(dim=0) # take marginal over generated samples to get mean classifier distribution
    clas_entropy = -(p*p.log()).sum(dim=-1) # (N,) in nats
    conditional_entropy = float(clas_entropy) - mutual_info
    logging.debug(f"{mutual_info}, {p}, {clas_entropy}, {conditional_entropy}")
    return math.exp(conditional_entropy)

    
