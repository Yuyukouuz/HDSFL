# -*- coding: utf-8 -*-
"""
@author: QgZhan
@contact: zhanqg@foxmail.com
@file: loss_function.py
@time: 2022/4/21 15:49
"""

# coding = utf-8

import torch
import numpy
from torch import nn, nn as nn


def gram_linear(x):
    """Compute Gram (kernel) matrix for a linear kernel.

    Args:
    x: A num_examples x num_features matrix of features.

    Returns:
    A num_examples x num_examples Gram matrix of examples.
    """
    return torch.mm(x, torch.t(x))


def gram_rbf(x, threshold=1.0):
    """Compute Gram (kernel) matrix for an RBF kernel.

    Args:
    x: A num_examples x num_features matrix of features.
    threshold: Fraction of median Euclidean distance to use as RBF kernel
      bandwidth. (This is the heuristic we use in the paper. There are other
      possible ways to set the bandwidth; we didn't try them.)

    Returns:
    A num_examples x num_examples Gram matrix of examples.
    """
    dot_products = torch.mm(x, torch.t(x))
    sq_norms = torch.diag(dot_products)
    sq_distances = -2 * dot_products + sq_norms[:, None] + sq_norms[None, :]
    sq_median_distance = torch.median(sq_distances)
    return torch.exp(-sq_distances / (2 * threshold ** 2 * sq_median_distance + 1e-7))


def center_gram(gram, unbiased=False):
    """Center a symmetric Gram matrix.

    This is equvialent to centering the (possibly infinite-dimensional) features
    induced by the kernel before computing the Gram matrix.

    Args:
    gram: A num_examples x num_examples symmetric matrix.
    unbiased: Whether to adjust the Gram matrix in order to compute an unbiased
      estimate of HSIC. Note that this estimator may be negative.

    Returns:
    A symmetric matrix with centered columns and rows.
    """
    # if not float(torch.max(torch.t(gram) - gram)) < 1e-6:
    #     raise ValueError('Input must be a symmetric matrix.')
    gram = gram.clone()

    if unbiased:
        # This formulation of the U-statistic, from Szekely, G. J., & Rizzo, M.
        # L. (2014). Partial distance correlation with methods for dissimilarities.
        # The Annals of Statistics, 42(6), 2382-2412, seems to be more numerically
        # stable than the alternative from Song et al. (2007).
        n = gram.shape[0]
        gram = gram - torch.diag(torch.diag(gram))
        means = torch.sum(gram, 0).float() / (n - 2)
        means -= torch.sum(means) / (2 * (n - 1))
        gram -= means[:, None]
        gram -= means[None, :]
        gram = gram - torch.diag(torch.diag(gram))
    else:
        means = torch.mean(gram, 0)
        means -= torch.mean(means) / 2
        gram -= means[:, None]
        gram -= means[None, :]

    return gram


def cka(gram_x, gram_y, debiased=False):
    """Compute CKA.

    Args:
    gram_x: A num_examples x num_examples Gram matrix.
    gram_y: A num_examples x num_examples Gram matrix.
    debiased: Use unbiased estimator of HSIC. CKA may still be biased.

    Returns:
    The value of CKA between X and Y.
    """
    gram_x = center_gram(gram_x, unbiased=debiased)
    gram_y = center_gram(gram_y, unbiased=debiased)

    # Note: To obtain HSIC, this should be divided by (n-1)**2 (biased variant) or
    # n*(n-3) (unbiased variant), but this cancels for CKA.
    scaled_hsic = torch.mm(gram_x.view(1, -1), torch.t(gram_y.view(1, -1)))

    normalization_x = torch.norm(gram_x)
    normalization_y = torch.norm(gram_y)
    return scaled_hsic / (normalization_x * normalization_y)


def linear_CKA(gram_x, gram_y, ftype="SUM", debiased=False):
    """Compute CKA.

    Args:
    gram_x: A num_examples x num_examples Gram matrix.
    gram_y: A num_examples x num_examples Gram matrix.
    ftype: The function type, which decide how to reshape the input, including:
           "SUM" -- sum the value along time dimension;  # (N, T, feature_dim) -> (N, feature_dim)
           "FLATTEN" -- faltten the time dimension;  # (N, T, feature_dim) -> (N, T * feature_dim)

    debiased: Use unbiased estimator of HSIC. CKA may still be biased.

    Returns:
    The value of CKA between X and Y.
    """
    if len(gram_x.shape) > 2 and len(gram_y) > 2:
        if ftype == "SUM":
            gram_x = torch.sum(gram_x, 1)  # (N, T, feature_dim) -> (N, feature_dim)
            gram_y = torch.sum(gram_y, 1)
        elif ftype == "FLATTEN":
            gram_x = torch.flatten(gram_x, 1)  # (N, T, feature_dim) -> (N, T * feature_dim)
            gram_y = torch.flatten(gram_y, 1)

    gram_x = center_gram(gram_linear(gram_x), unbiased=debiased)
    gram_y = center_gram(gram_linear(gram_y),  unbiased=debiased)
    # Note: To obtain HSIC, this should be divided by (n-1)**2 (biased variant) or
    # n*(n-3) (unbiased variant), but this cancels for CKA.
    scaled_hsic = torch.mm(gram_x.view(1, -1), torch.t(gram_y.view(1, -1)))
    normalization_x = torch.norm(gram_x)
    normalization_y = torch.norm(gram_y)
    return scaled_hsic / (normalization_x * normalization_y + 1e-7)


def rbf_CKA(gram_x, gram_y, ftype="SUM", debiased=False):
    """Compute CKA.

    Args:
    gram_x: A num_examples x num_examples Gram matrix.
    gram_y: A num_examples x num_examples Gram matrix.
    debiased: Use unbiased estimator of HSIC. CKA may still be biased.

    Returns:
    The value of CKA between X and Y.
    """
    if len(gram_x.shape) > 2 and len(gram_y) > 2:
        if ftype == "SUM":
            gram_x = torch.sum(gram_x, 1)  # (N, T, feature_dim) -> (N, feature_dim)
            gram_y = torch.sum(gram_y, 1)
        elif ftype == "FLATTEN":
            gram_x = torch.flatten(gram_x, 1)  # (N, T, feature_dim) -> (N, T * feature_dim)
            gram_y = torch.flatten(gram_y, 1)

    gram_x = center_gram(gram_rbf(gram_x), unbiased=debiased)
    gram_y = center_gram(gram_rbf(gram_y), unbiased=debiased)

    # Note: To obtain HSIC, this should be divided by (n-1)**2 (biased variant) or
    # n*(n-3) (unbiased variant), but this cancels for CKA.
    scaled_hsic = torch.mm(gram_x.view(1, -1), torch.t(gram_y.view(1, -1)))

    normalization_x = torch.norm(gram_x)
    normalization_y = torch.norm(gram_y)
    return scaled_hsic / (normalization_x * normalization_y)


def temporal_linear_CKA(gram_x, gram_y, debiased=False):
    """Compute temporal linear CKA.

    Args:
    gram_x: A num_examples x num_examples Gram matrix.
    gram_y: A num_examples x num_examples Gram matrix.
    debiased: Use unbiased estimator of HSIC. CKA may still be biased.

    Returns:
    The value of CKA between X and Y.
    """
    loss = 0
    for t in range(gram_x.shape[1]):
        loss += linear_CKA(gram_x[:, t, :], gram_y[:, t, :], debiased)
    return loss / gram_x.shape[1]


def temporal_rbf_CKA(gram_x, gram_y, debiased=False):
    """Compute temporal rbf CKA.

    Args:
    gram_x: A num_examples x num_examples Gram matrix.
    gram_y: A num_examples x num_examples Gram matrix.
    debiased: Use unbiased estimator of HSIC. CKA may still be biased.

    Returns:
    The value of CKA between X and Y.
    """
    loss = 0
    for t in range(gram_x.shape[1]):
        loss += rbf_CKA(gram_x[:, t, :], gram_y[:, t, :], debiased)
    return loss / gram_x.shape[1]


def _debiased_dot_product_similarity_helper(
        xty, sum_squared_rows_x, sum_squared_rows_y, squared_norm_x, squared_norm_y,
        n):
    """Helper for computing debiased dot product similarity (i.e. linear HSIC)."""
    # This formula can be derived by manipulating the unbiased estimator from
    # Song et al. (2007).
    return (
            xty - n / (n - 2.) * torch.sum(sum_squared_rows_x * sum_squared_rows_y)
            + squared_norm_x * squared_norm_y / ((n - 1) * (n - 2)))


def feature_space_linear_cka(features_x, features_y, debiased=False):
    """Compute CKA with a linear kernel, in feature space.

    This is typically faster than computing the Gram matrix when there are fewer
    features than examples.

    Args:
    features_x: A num_examples x num_features matrix of features.
    features_y: A num_examples x num_features matrix of features.
    debiased: Use unbiased estimator of dot product similarity. CKA may still be
      biased. Note that this estimator may be negative.

    Returns:
    The value of CKA between X and Y.
    """
    features_x = features_x - torch.mean(features_x, 0, True)
    features_y = features_y - torch.mean(features_y, 0, True)

    dot_product_similarity = torch.norm(torch.mm(torch.t(features_x), features_y)) ** 2
    normalization_x = torch.norm(torch.mm(torch.t(features_x), features_x))
    normalization_y = torch.norm(torch.mm(torch.t(features_y), features_y))

    if debiased:
        n = features_x.shape[0]
        # Equivalent to np.sum(features_x ** 2, 1) but avoids an intermediate array.
        sum_squared_rows_x = torch.einsum('ij,ij->i', features_x, features_x)
        sum_squared_rows_y = torch.einsum('ij,ij->i', features_y, features_y)
        squared_norm_x = torch.sum(sum_squared_rows_x)
        squared_norm_y = torch.sum(sum_squared_rows_y)

        dot_product_similarity = _debiased_dot_product_similarity_helper(
            dot_product_similarity, sum_squared_rows_x, sum_squared_rows_y,
            squared_norm_x, squared_norm_y, n)
        normalization_x = torch.sqrt(_debiased_dot_product_similarity_helper(
            normalization_x ** 2, sum_squared_rows_x, sum_squared_rows_x,
            squared_norm_x, squared_norm_x, n))
        normalization_y = torch.sqrt(_debiased_dot_product_similarity_helper(
            normalization_y ** 2, sum_squared_rows_y, sum_squared_rows_y,
            squared_norm_y, squared_norm_y, n))

    return dot_product_similarity / (normalization_x * normalization_y)


def Entropy(input_):
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy


def grl_hook(coeff):
    def fun1(grad):
        return -coeff * grad.clone()

    return fun1


def CDAN(input_list, ad_net, entropy=None, coeff=None, random_layer=None):
    softmax_output = input_list[1].detach()
    feature = input_list[0]
    if random_layer is None:
        op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
        op_out.requires_grad_()
        ad_out = ad_net(op_out.view(-1, softmax_output.size(1) * feature.size(1)))
    else:
        random_out = random_layer.forward([feature, softmax_output])
        random_out.requires_grad_()
        ad_out = ad_net(random_out.view(-1, random_out.size(1)))
    batch_size = softmax_output.size(0) // 2
    dc_target = torch.from_numpy(numpy.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
    if entropy is not None:
        entropy.requires_grad_()
        entropy.register_hook(grl_hook(coeff))
        entropy = 1.0 + torch.exp(-entropy)
        source_mask = torch.ones_like(entropy)
        source_mask[feature.size(0) // 2:] = 0
        source_weight = entropy * source_mask
        target_mask = torch.ones_like(entropy)
        target_mask[0:feature.size(0) // 2] = 0
        target_weight = entropy * target_mask
        weight = source_weight / torch.sum(source_weight).detach().item() + \
                 target_weight / torch.sum(target_weight).detach().item()
        l = nn.BCELoss(reduction='none')(ad_out, dc_target)
        return torch.sum(weight.view(-1, 1) * nn.BCELoss()(ad_out, dc_target)) / torch.sum(weight).detach().item()
    else:
        return nn.BCELoss()(ad_out, dc_target)


def mdd_loss(features, labels, left_weight=1, right_weight=1):
    softmax_out = nn.Softmax(dim=1)(features)
    batch_size = features.size(0)
    if float(batch_size) % 2 != 0:
        raise Exception('Incorrect batch size provided')

    batch_left = softmax_out[:int(0.5 * batch_size)]
    batch_right = softmax_out[int(0.5 * batch_size):]

    loss = torch.norm((batch_left - batch_right).abs(), 2, 1).sum() / float(batch_size)

    labels_left = labels[:int(0.5 * batch_size)]
    batch_left_loss = get_pari_loss1(labels_left, batch_left)

    labels_right = labels[int(0.5 * batch_size):]
    batch_right_loss = get_pari_loss1(labels_right, batch_right)
    return loss + left_weight * batch_left_loss + right_weight * batch_right_loss


def mdd_digit(features, labels, left_weight=1, right_weight=1, weight=1):
    softmax_out = nn.Softmax(dim=1)(features)
    batch_size = features.size(0)
    if float(batch_size) % 2 != 0:
        raise Exception('Incorrect batch size provided')

    batch_left = softmax_out[:int(0.5 * batch_size)]
    batch_right = softmax_out[int(0.5 * batch_size):]

    loss = torch.norm((batch_left - batch_right).abs(), 2, 1).sum() / float(batch_size)

    labels_left = labels[:int(0.5 * batch_size)]
    labels_left_left = labels_left[:int(0.25 * batch_size)]
    labels_left_right = labels_left[int(0.25 * batch_size):]

    batch_left_left = batch_left[:int(0.25 * batch_size)]
    batch_left_right = batch_left[int(0.25 * batch_size):]
    batch_left_loss = get_pair_loss(labels_left_left, labels_left_right, batch_left_left, batch_left_right)

    labels_right = labels[int(0.5 * batch_size):]
    labels_right_left = labels_right[:int(0.25 * batch_size)]
    labels_right_right = labels_right[int(0.25 * batch_size):]

    batch_right_left = batch_right[:int(0.25 * batch_size)]
    batch_right_right = batch_right[int(0.25 * batch_size):]
    batch_right_loss = get_pair_loss(labels_right_left, labels_right_right, batch_right_left, batch_right_right)

    return weight*loss + left_weight * batch_left_loss + right_weight * batch_right_loss


def get_pair_loss(labels_left, labels_right, features_left, features_right):
    loss = 0
    for i in range(len(labels_left)):
        if (labels_left[i] == labels_right[i]):
            loss += torch.norm((features_left[i] - features_right[i]).abs(), 2, 0).sum()
    return loss


def get_pari_loss1(labels, features):
    loss = 0
    count = 0
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if (labels[i] == labels[j]):
                count += 1
                loss += torch.norm((features[i] - features[j]).abs(), 2, 0).sum()
    return loss / count


def EntropicConfusion(features):
    softmax_out = nn.Softmax(dim=1)(features)
    batch_size = features.size(0)
    loss = torch.mul(softmax_out, torch.log(softmax_out)).sum() * (1.0 / batch_size)
    return loss


def TET_loss(outputs, labels, criterion=nn.CrossEntropyLoss(), means=1.0, lamb=1e-3):
    """
    calculate TET loss.
    $L_{TET} = \frac{1}{T} \cdot \sum_{t=1}^{T} L_{CE}[O(t), y]$
    This code is form https://github.com/Gus-Lab/temporal_efficient_training
    :param outputs: the predict results
    :param labels: the true labels
    :param criterion: the criterion in TET loss
    :param means:  the target of regularization loss.
    :param lamb: the ratio of regularization loss, which aims to avoid the outputs occur outliers.
    :return:
    """
    T = outputs.size(1)
    loss_es = 0
    for t in range(T):
        loss_es += criterion(outputs[:, t, ...], labels)
    loss_es = loss_es / T  # L_TET
    if lamb != 0:
        MSE_loss = torch.nn.MSELoss()
        y = torch.zeros_like(outputs).fill_(means)
        regularization_loss = MSE_loss(outputs, y)  # L_mse
    else:
        regularization_loss = 0
    return (1 - lamb) * loss_es + lamb * regularization_loss  # L_Total