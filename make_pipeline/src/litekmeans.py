#!/usr/bin/env python
#-*- coding: utf-8 -*-

from __future__ import division
import time

import numpy as np
import scipy.sparse
import handythread


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


def litekmeans(X, k, max_iter=50):
    X = X.T
    n = X.shape[1]
    ndim = X.shape[0]
    last = 0
    label = np.random.randint(k, size=(n, ))
    iteration = 0
    batchsize = 100000
    nbatches = int(np.ceil(n / batchsize))
    center = np.zeros((ndim, k), dtype=np.float32)
    while np.any(label != last):
        start = time.time()
        iteration += 1
        print 'iteration: {0}'.format(iteration)

        E = scipy.sparse.coo_matrix(
            (np.ones((n, ), dtype=np.int), (np.arange(n), label)), shape=(n, k), dtype=np.float64).tocsr()

        # E = one hot assignments
        # spdiags... = counts
        print 'max of E.sum(0): %s' % (E.sum(0).max(), )
        print 'max of (1.0/E.sum(0)): %s' % ((1.0/E.sum(0)).max(), )
        print 'min of E.sum(0): %s' % (E.sum(0).min(), )
        print 'min of (1.0/E.sum(0)): %s' % ((1.0/E.sum(0)).min(), )
        print 'np.all(1.0/E.sum(0) == np.inf): %r' % (np.all(1.0/E.sum(0) == np.inf), )
        center = X * E * scipy.sparse.spdiags(1.0 / (E.sum(0)+0.0000000001), 0, k, k)
        c2 = 0.5 * np.sum(center ** 2, 0).T[:, None]
        last = label
        label = np.zeros((n, ), dtype=np.int)

        def get_labels(batchidx):
            return np.argmax(
                np.dot(center.T,
                        X[:,
                            j * batchsize + batchidx * 1000:min(
                                n, j * batchsize + (batchidx + 1) * 1000)]) -
                c2, axis=0)
        for j in range(nbatches):
            print 'processing batch {0:d} / {1:d}'.format(j + 1, nbatches)


            tmp = handythread.parallel_map(get_labels, range(int(np.ceil(batchsize / 1000))), threads=8)
            label[j * batchsize: min(n, int((j + 1) * batchsize))] = np.concatenate(tmp)
        if iteration >= max_iter:
            break
        print 'iteration took {0:d} seconds'.format(int(time.time() - start))
    obj = 0
    Xsq = 0.5 * np.sum(X ** 2, 0)
    batchsize = 10000
    nbatches = int(np.ceil(n / batchsize))
    csq = 0.5 * np.sum(center ** 2, 0)
    # TODO: do this stuff in parallel as well (takes longer than expected)
    def compute_sqd(batchidx):
        tempX = X[:, j * batchsize + batchidx * 100:min(n, j * batchsize + (batchidx + 1) * 100)]
        temp = np.dot(-center.T, tempX) + csq[:, None]
        tmp = Xsq[j * batchsize + batchidx * 100:min(n, j * batchsize + (batchidx + 1) * 100)] + temp
        temp_mindist = np.min(
            Xsq[j * batchsize + batchidx * 100:min(n, j * batchsize + (batchidx + 1) * 100)] + temp,
            axis=0
        )
        return np.sum(temp_mindist)
    for j in range(nbatches):
        tmp = handythread.parallel_map(compute_sqd, range(int(np.ceil(batchsize / 100))), threads=8)
        obj += np.sum(tmp)
        print 'obj: %r' % (obj, )
    #print obj
    #for j in range(nbatches):
    #    tempX = X[:, j * batchsize:min(n, (j + 1) * batchsize)]
    #    temp = np.dot(-center.T, tempX) + csq[:, None]
    #    print 'Xsq[j * batchsize:min(n, (j + 1) * batchsize)].mean(): %r' % (
    #        Xsq[j * batchsize:min(n, (j + 1) * batchsize)].mean(), )
    #    print 'mean of temp: %s' % (temp.mean(), )
    #    tmp = Xsq[j * batchsize:min(n, (j + 1) * batchsize)] + temp
    #
    #    temp_mindist = np.min(
    #        Xsq[j * batchsize:min(n, (j + 1) * batchsize)] + temp,
    #        axis=0
    #    )
    #    obj = obj + np.sum(temp_mindist)
    print 'obj: %r' % (obj, )
    center = center.T
    return (label, center, obj)


if __name__ == '__main__':
#    comm = MPI.COMM_WORLD
#    if comm.rank == 0:
    iterations = 50
    k = 3000
    data = np.load('/home/vincent/data/mappings_3000000x512.npy')[:3000000]
    start = time.time()
    label, centroids, obj = litekmeans(data, k, iterations)
    #tmp,tmp2 = litekmeans(data, k, iterations)
    #pprint('clustering took %f seconds' % (time.time() - start, ))
    np.save('/home/vincent/data/testcentroids.npy', centroids)
    print 'time needed: %f' % (time.time() - start, )


# vim: set ts=4 sw=4 sts=4 expandtab:
