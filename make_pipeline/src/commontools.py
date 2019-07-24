#!/usr/bin/env python
#-*- coding: utf-8 -*-

import errno
import glob
import os

import numpy as np

def sparsify_weights(weights, n_in, scale, scale_small=0.0):
    """sparsifies weight matrices and rescales weights
    Source:
    http://www.cs.utoronto.ca/~ilya/rnn_code.tar.gz

    Args:
        weights: weight matrix
        n_in: number of big incoming weights per unit
        scale: scale factor of the big weights
        scale_small: scale factor for the small weights, defaults to 0

    returns
    """
    assert type(n_in)==int
    W = weights * scale

    for i in range(W.shape[1]):
        perm = np.random.permutation(W.shape[1])
        SMALL = perm[n_in:]
        W[SMALL, i] *= scale_small / scale

    # update weights in-place
    weights[:] = W

def get_new_path_idx(parent_folder, path_prefix, digits=5):
    """Checks directory names with suffix num and returns the highest number+1
    Can be useful for auto-incrementing output directory names
    Args:
        parent_folder: directory in which we check the filenames
        path_prefix: part of the filename before the number
        digits: number of digits in the filenames
    Returns:
        Number
    """
    newidx = 0
    print parent_folder, path_prefix, digits
    for f in glob.glob('{0}/{1}{2}'.format(parent_folder, path_prefix, '[0-9]' * digits)):
        idx = int(f[-digits:])
        if idx >= newidx:
            newidx = idx + 1
    if newidx >= 10**(digits - 1):
        raise IndexError('new index cannot be represented by {0} digits'.format(digits))
    return newidx


def pformat_sorteddict(d, indentspaces=4, indentlevel=1):
    """Generates a pretty string of a nested dictionary with sorted items
    Args:
        d: the dictionary
        indentspaces: number of spaces per indentation level
        indentlevel: level of indentation to begin with (>=1)
    Returns:
        The formatted string
    """
    d_str = '{0}{{'.format(' ' * indentspaces * (indentlevel - 1))
    for k, v in sorted(d.items()):
        d_str += '{0}{1}{2}:'.format(os.linesep, ' ' * indentspaces * indentlevel, str(k))
        if isinstance(v, dict):
            d_str += pprint_sorteddict(v, indentspaces, indentlevel + 1)
        else:
            d_str += '{0}{1}{2}'.format(os.linesep, ' ' * indentspaces * (indentlevel + 1), str(v))
        d_str += ',{0}'.format(os.linesep)
    d_str += '{0}}}'.format(' ' * indentspaces * (indentlevel - 1))
    print d_str


def pprint_sorteddict(d, indentspaces=4, indentlevel=1):
    """Pretty prints a nested dictionary with sorted items
    Args:
        d: the dictionary
        indentspaces: number of spaces per indentation level
        indentlevel: level of indentation to begin with (>=1)
    """
    d_str = '{0}{{'.format(' ' * indentspaces * (indentlevel - 1))
    for k, v in sorted(d.items()):
        d_str += '{0}{1}{2}:'.format(os.linesep, ' ' * indentspaces * indentlevel, str(k))
        if isinstance(v, dict):
            d_str += pprint_sorteddict(v, indentspaces, indentlevel + 1)
        else:
            d_str += '{0}{1}{2}'.format(os.linesep, ' ' * indentspaces * (indentlevel + 1), str(v))
        d_str += ',{0}'.format(os.linesep)
    d_str += '{0}}}'.format(' ' * indentspaces * (indentlevel - 1))
    print d_str


def saferepr_sorteddict(d, indentspaces=4, indentlevel=1):
    """Generates representation string of nested dictionary with sorted items
    Args:
        d: the dictionary
        indentspaces: number of spaces per indentation level
        indentlevel: level of indentation to begin with (>=1)
    Returns:
        The formatted representation string
    """
    d_repr = '{0}{{'.format(' ' * indentspaces * (indentlevel - 1))
    for k, v in sorted(d.items()):
        d_repr += '{0}{1}{2}:'.format(os.linesep, ' ' * indentspaces * indentlevel, repr(k))
        if isinstance(v, dict):
            d_repr += saferepr_sorteddict(v, indentspaces, indentlevel + 1)
        else:
            d_repr += '{0}{1}{2}'.format(os.linesep, ' ' * indentspaces * (indentlevel + 1), repr(v))
        d_repr += ',{0}'.format(os.linesep)
    d_repr += '{0}}}'.format(' ' * indentspaces * (indentlevel - 1))
    return d_repr


def make_sure_path_exists(path):
    """Checks if path exists, if not creates it
    Args:
        path: the path that should be checked
    Raises:
        OSError: if directory does not exist, but os.makedirs() fails anyway
    """
    try:
        os.makedirs(path)
        os.chmod(path, 0700)
        print 'created new path %s' % (path, )
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise OSError('Error creating directory, although it didn\'t exist.')

def norm_along_last_axis(x):
    """Calculates the norm along the last axis

    Args:
        x: numpy array
    Returns:
        Numpy array containing euclidean norm for vectors on the last axis of x.
    """
    return (np.sum(np.abs(x)**2, axis=-1)**(1./2))[..., None]


def onehot(x, numclasses=None):
    if x.shape == ():
        x = x[None]
    if numclasses is None:
        numclasses = x.max() + 1
    result = np.zeros(list(x.shape) + [numclasses])
    z = np.zeros(x.shape)
    for c in range(numclasses):
        z *= 0
        z[np.where(x == c)] = 1
        result[..., c] += z
    return result


# vim: set ts=4 sw=4 sts=4 expandtab:
