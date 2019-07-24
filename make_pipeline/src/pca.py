#!/usr/bin/env python
#-*- coding: utf-8 -*-


import numpy as np
import theano
import theano.tensor as T

from theano_tools import (theano_subtract_row, theano_divide_row,
                          theano_mean0, theano_matrix1, theano_matrix2,
                          theano_row_vector, theano_scalar, theano_dot)


_theano_std0_sum = T.sum((theano_matrix1 - theano_row_vector)**2 / (theano_scalar-1), # Bessel's correction
                         0)
theano_subtract_m1 = theano.function(
    inputs=[theano_matrix1],
    outputs=theano_matrix1 - T.mean(theano_matrix1, 1).dimshuffle(0, 'x'))

_s1 = T.std(theano_matrix1, 1).dimshuffle(0, 'x')
theano_divide_s1 = theano.function(
    inputs=[theano_matrix1],
    outputs=theano_matrix1 / (_s1 + _s1.mean()))


#NOTE: BE CAREFUL TO REUSE num_processed_cases only in the corresponding function!!!! bad example: feeding num_processed_cases returned by compute_mean0_iteratively to compute_cov_iteratively.
def compute_mean0_iteratively(batch, use_gpu=True, m0=None, nproccases=0):
    """Computes the mean over vertically stacked data points iteratively.

    This function can be used to compute the mean over data points online.

    Args:
        batch: a batch of data
        use_gpu: Indicates, whether theano should be used or not.
        m0: if this is not the first iteration, this should hold the result of
            the previous iteration
        num_processed_cases: this should hold the number of processed data
            points up to including the last iteration
    Returns:
        A tuple containing the mean over data points and the number of
        processed cases.
    """
    if m0 is None:
        if use_gpu:
            return theano_mean0(batch).reshape((1, -1)), batch.shape[0]
        else:
            return batch.mean(0).reshape((1, -1)), batch.shape[0]
    else:
        w0 = np.float32(nproccases) / (nproccases + batch.shape[0])
        nproccases += batch.shape[0]
        if use_gpu:
            return w0 * m0 + (1-w0) * theano_mean0(batch), nproccases
        else:
            return w0 * m0 + (1-w0) * batch.mean(0), nproccases

def compute_mean0_batchwise(data, batchsize=100, use_gpu=True, verbose=True):
    """Computes the mean over vertically stacked data points batchwise.

   This function partitions the data into batches and updates the mean batch
    after batch.

    Args:
        data: The vertically stacked data points
        batchsize: The number of data points in each batch
        use_gpu: Indicates, whether theano should be used or not.
        verbose: Set to True for debug output
    Returns:
        The mean over data points.
    """
    ncases, ndim = data.shape
    nbatches = (ncases - 1) / batchsize + 1
    m0 = np.zeros((1, ndim), dtype=theano.config.floatX)
    nproccases = 0
    for bidx in range(nbatches):
        if verbose:
            print "processing batch %d of %d" % (bidx+1, nbatches)
        start = bidx * batchsize
        end = min((bidx + 1) * batchsize, ncases)
        m0[:, :], nproccases = compute_mean0_iteratively(
            batch=data[start:end], use_gpu=use_gpu, m0=m0, nproccases=nproccases)
    return m0

def compute_std0_batchwise(data, use_gpu=True, batchsize=100, verbose=True):
    """Computes the standard deviation over stacked data points batchwise.

    This function partitions the data into batches, updates the variance
    batch after batch and then computes the square root of the variance.

    Args:
        data: The vertically stacked data points
        use_gpu: Indicates, whether theano should be used or not.
        batchsize: The number of data points in each batch
    Returns:
        The standard deviation over stacked data points.
    """
    ncases, ndim = data.shape
    nbatches = (ncases - 1) / batchsize + 1
    m0 = compute_mean0_batchwise(data, use_gpu=True, batchsize=batchsize)
    if use_gpu:
        s0 = theano.shared(np.zeros((1, ndim), dtype=theano.config.floatX), name='s0')
        s0_update_f = theano.function(inputs=[theano_matrix1, theano_row_vector, theano_scalar],
                                      outputs=[],
                                      updates={s0: s0 + _theano_std0_sum})
    else:
        s0 = np.zeros((1, data.shape[1]), dtype=theano.config.floatX)
    if data.shape[0] < batchsize:
        batchsize = data.shape[0]
    for bidx in range(nbatches):
        if verbose:
            print "processing batch %d of %d" % (bidx+1, nbatches)
        start = bidx * batchsize
        end = min((bidx + 1) * batchsize, ncases)
        if use_gpu:
            s0_update_f(data[start:end], m0, ncases)
        else:
            s0[:, :] += np.sum((data[start:end] - m0)**2 / (ncases-1), axis=0)

    if use_gpu:
        return np.sqrt(s0.get_value())
    else:
        return np.sqrt(s0)

def compute_covmat_batchwise(data, use_gpu=True, batchsize=100, verbose=True):
    """Computes the covariance matrix for vertically stacked data points.

    Args:
        data: The vertically stacked data points.
        use_gpu: Indicates, whether theano should be used or not.
        batchsize: The number of data points in each batch
        verbose: Set to True for debug output
    Returns:
        The covariance matrix for the data.
    """
    ncases, ndim = data.shape
    nbatches = (ncases - 1) / batchsize + 1
    if use_gpu:
        if verbose:
            print 'using gpu'
        C = theano.shared(np.zeros((ndim, ndim),
                                   dtype=theano.config.floatX), name='C')
        update_C_f = theano.function(inputs=[theano_matrix1, theano_matrix2, theano_scalar],
                                      outputs=[],
                                      updates={
                                          C: C + T.dot(theano_matrix1, theano_matrix2) / theano_scalar
                                      })
    else:
        C = np.zeros((ndim, ndim), dtype=theano.config.floatX)
    for bidx in range(nbatches):
        if verbose:
            print "processing batch %d of %d" % (bidx+1, nbatches)
        start = bidx * batchsize
        end = min((bidx + 1) * batchsize, ncases)
        if use_gpu:
            update_C_f(data.T[:, start:end],
                       data[start:end, :],
                       np.float32(ncases))
        else:
            C += np.dot(data.T[:, start:end],
                       data[start:end, :]) / np.float32(ncases)
    if use_gpu:
        return C.get_value()
    else:
        return C

def _get_pca_params_from_covmat(C, verbose=True):
    """Computes pca parameters from the given data covariance matrix.

    Args:
        C: The data covariance matrix
        verbose: Set to True for debug output
    Returns:
        A tuple containing the PCA-Matrix, the inverse PCA Matrix and an array
        of fractions of variance (how much of the variance will be retained, if
        all rows after this one are dropped from V).
    """
    if verbose:
        print 'done'
        print 'calculating eigenvalue decomposition'

    u, v = np.linalg.eigh(C)
    v = v[:, np.argsort(u)[::-1]]
    u.sort()
    u = u[::-1]
    u = u[u.cumsum()<u.sum()*(0.99999)] # throw away some eigenvalues for numerical stability
    var_fracs = u.cumsum()/u.sum()
    numprincomps = u.shape[0]
    V = ((u**(-0.5))[:numprincomps][np.newaxis, :]*v[:, :numprincomps]).T
    W = (u**0.5)[:numprincomps][np.newaxis, :]*v[:, :numprincomps]
    if verbose:
        print 'done'

    return V, W, var_fracs

def pca(data, use_gpu = True, batchsize=100, const_norm=True, data_norm=True, verbose=True):
    data = data.astype(np.float32)
    ncases = data.shape[0]
    
    nbatches = (ncases - 1) / batchsize + 1
    if const_norm:
        # contrast normalization
        for bidx in range(nbatches):
            start = bidx * batchsize
            end = min((bidx + 1) * batchsize, ncases)
            if use_gpu:
                data[start:end] = theano_subtract_m1(data[start:end])
                data[start:end] = theano_divide_s1(data[start:end])
            else:
                data[start:end] -= data[start:end].mean(1)[:, None]
                s1 = data[start:end].std(1)[:, None]
                data[start:end] /= s1 + s1.mean()
    # normalization over dataset
    if data_norm:
        m0 = compute_mean0_batchwise(data, batchsize=batchsize, use_gpu=use_gpu, verbose=verbose)
        for bidx in range(nbatches):
            start = bidx * batchsize
            end = min((bidx + 1) * batchsize, ncases)
            if use_gpu:
                data[start:end] = theano_subtract_row(data[start:end], m0)
            else:
                data[start:end] -= m0

        s0 = compute_std0_batchwise(data, batchsize=batchsize, use_gpu=use_gpu, verbose=verbose)
        s0 += s0.mean()
        for bidx in range(nbatches):
            start = bidx * batchsize
            end = min((bidx + 1) * batchsize, ncases)
            if use_gpu:
                data[start:end] = theano_divide_row(data[start:end], s0)
            else:
                data[start:end] /= s0
    else:
        m0 = 0
        s0 = 0

    covmat = compute_covmat_batchwise(data, use_gpu=use_gpu, batchsize=batchsize, verbose=verbose)
    V, W, var_fracs = _get_pca_params_from_covmat(covmat, verbose=verbose)
    return V, W, m0, s0, var_fracs

def whiten(data, V, m0, s0, var_fracs, retain_var, nprincomps=0, batchsize=1000, use_gpu=True, verbose=True):
    data = data.astype(np.float32)
    ncases = data.shape[0]
    nbatches = (ncases - 1) / batchsize + 1
    if nprincomps == 0:
        nprincomps = np.where(var_fracs > retain_var)[0][0]
    data_white = np.zeros((ncases, nprincomps), dtype=np.float32)
    for bidx in range(nbatches):
        start = bidx * batchsize
        end = min((bidx + 1) * batchsize, ncases)
        data[start:end] -= data[start:end].mean(1)[:, None]
        s1 = data[start:end].std(1)[:, None]
        data[start:end] /= s1 + s1.mean()
        data[start:end] -= m0
        data[start:end] /= s0
        if use_gpu:
            data_white[start:end] = theano_dot(data[start:end], V[:nprincomps].T)
        else:
            data_white[start:end] = np.dot(data[start:end], V[:nprincomps].T)
    return data_white


if __name__ == '__main__':
    import tables
    data = np.load('/home/shared/data/hollywood2/l2_inputs_500000x8x300.npy', 'r')
    data = data.reshape(-1,2400)
    ncases = data.shape[0]
    h5file = tables.openFile(
        '/home/shared/data/hollywood2/layer2_input_traindata_.h5',
        mode='w', title='input data for layer2 (conc and whitened)')
    preprocessing_params = np.load(
        '/home/shared/data/hollywood2/l2_preprocessing_params.npz')

    retain_var = 0.8491
    nprincomps = np.where(preprocessing_params['var_fracs'] > retain_var)[0][0]
    print 'components retained',nprincomps
    retained_var = preprocessing_params['var_fracs'][nprincomps]
    h5file.createGroup('/', 'preprocessing_params',
                       'PCA Whitening Parameters')
    h5file.createArray(h5file.root.preprocessing_params,
                       'V', preprocessing_params['V'].astype(np.float32))
    h5file.createArray(h5file.root.preprocessing_params,
                       'W', preprocessing_params['W'].astype(np.float32))
    h5file.createArray(h5file.root.preprocessing_params,
                       'm0', preprocessing_params['m0'].astype(np.float32))
    h5file.createArray(h5file.root.preprocessing_params,
                       's0', preprocessing_params['s0'].astype(np.float32))
    h5file.createArray(h5file.root.preprocessing_params,
                       'var_fracs', preprocessing_params['var_fracs'].astype(np.float32))
    h5file.createArray(h5file.root.preprocessing_params,
                       'retained_var', retained_var)
    h5file.createArray(h5file.root.preprocessing_params,
                       'nprincomps', nprincomps)
    h5file.createCArray(h5file.root, 'data_white', tables.Float32Atom(),
                        shape=(ncases, nprincomps),
                        chunkshape=(100000, nprincomps))
    batchsize = 100000
    nbatches = (ncases - 1) / batchsize + 1
    for bidx in range(nbatches):
        print "processing batch %d of %d" % (bidx+1, nbatches)
        start = bidx * batchsize
        end = min((bidx + 1) * batchsize, ncases)
        h5file.root.data_white[start:end] = whiten(
            data[start:end], preprocessing_params['V'].astype(np.float32),
            preprocessing_params['m0'].astype(np.float32),
            preprocessing_params['s0'].astype(np.float32),
            preprocessing_params['var_fracs'].astype(np.float32),
            retain_var,
            batchsize=10000, use_gpu=True, verbose=True)
        print "flushing..."
        h5file.flush()
    print 'done'
    h5file.close()


# vim: set ts=4 sw=4 sts=4 expandtab:
