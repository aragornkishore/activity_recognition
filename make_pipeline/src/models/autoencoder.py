#!/usr/bin/env python
#-*- coding: utf-8 -*-

from __future__ import division

import numpy as np

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

SMALL = 0.000001


class DenoisingAutoencoder(object):

    def __init__(self, n_in, n_hid, n_out, numpy_rng=None,
                 in_corruption_type='zeromask', in_corruption_level=0.5):
        theano.config.compute_test_value = 'warn'
        if numpy_rng is None:
            numpy_rng = np.random.RandomState(1)
        self.numpy_rng = numpy_rng
        self.theano_rng = RandomStreams(1)
        self.n_in = n_in
        self.n_out = n_out
        self.in_corruption_type = in_corruption_type
        self.in_corruption_level = in_corruption_level
        self.inputs = T.fmatrix(name='inputs')
        self.inputs.tag._test_value = np.random.randn(
            10, n_in).astype(theano.config.floatX)
        if isinstance(n_hid, int):
            self.n_hid = [n_hid]
        elif isinstance(n_hid, list) or isinstance(n_hid, tuple):
            self.n_hid = n_hid
        self.n_hid_layers = len(self.n_hid)

        self._init_params()

        # add params that should be trained to the params list
        self.params = [self.w_in_hid, self.w_hid_out,
                        self.b_out, self.b_in]
        self.params.extend(self.b_hid)
        self.params.extend(self.w_hid_hid)
        self.params.extend(self.b_hid_reconstructed)

        self._init_functions()

    def _init_params(self):
        # init input to hidden weights and biases
        w_in_hid_init = self.numpy_rng.uniform(
            low=-1/self.n_in, high=1/self.n_in,
            size=(self.n_in, self.n_hid[0])).astype(theano.config.floatX)
        self.w_in_hid = theano.shared(w_in_hid_init, name='w_in_hid')
        self.b_hid = []
        self.b_hid.append(
            theano.shared(np.zeros(self.n_hid[0], dtype=theano.config.floatX),
                          name='b_hid'))
        self.w_hid_hid = []
        # init multiple hid layers
        for i in range(1, self.n_hid_layers):
            w_hid_hid_init = self.numpy_rng.uniform(
                low=-1/self.n_hid[i-1], high=1/self.n_hid[i-1],
                size=(self.n_hid[i-1], self.n_hid[i])).astype(theano.config.floatX)
            self.w_hid_hid.append(theano.shared(
                w_hid_hid_init,
                name='w_h{}_h{}'.format(i-1, i)))
            self.b_hid.append(theano.shared(
                np.zeros(self.n_hid[i],
                         dtype=theano.config.floatX), name='b_h{}'.format(i)))

        # init hidden to output weights
        w_hid_out_init = self.numpy_rng.uniform(
            low=-1/self.n_hid[-1], high=1/self.n_hid[-1],
            size=(self.n_hid[-1], self.n_out)).astype(theano.config.floatX)
        self.w_hid_out = theano.shared(w_hid_out_init, name='w_hid_out')
        self.b_out = theano.shared(np.zeros(self.n_out, dtype=theano.config.floatX),
                                   name='b_out')
        self.b_in = theano.shared(np.zeros(self.n_in, dtype=theano.config.floatX),
                                  name='b_in')
        self.b_hid_reconstructed = []
        for i in range(self.n_hid_layers):
            self.b_hid_reconstructed.append(
                theano.shared(np.zeros(self.n_hid[i], dtype=theano.config.floatX),
                              name='b_h{}'.format(i)))

    def _init_functions(self):
        if self.in_corruption_type == 'zeromask':
            self._in_corrupted = self.theano_rng.binomial(
                size=self.inputs.shape, n=1, p=1.0-self.in_corruption_level,
                dtype=theano.config.floatX) * self.inputs

        self._hiddens = []
        self._hiddens.append(
            T.nnet.sigmoid(
                T.dot(self._in_corrupted, self.w_in_hid) + self.b_hid[0]))
        for i in range(1, self.n_hid_layers):
            self._hiddens.append(
            T.nnet.sigmoid(
                T.dot(self._hiddens[i-1], self.w_hid_hid[i-1]) + self.b_hid[i])
            )
        self._outputs = T.nnet.sigmoid(
            T.dot(self._hiddens[-1], self.w_hid_out) + self.b_out)
        self.outputs = theano.function(inputs=[self.inputs],
                                       outputs=self._outputs)
        self._hidden_reconstructed = []
        self._hidden_reconstructed.insert(
            0, T.nnet.sigmoid(
                T.dot(self._outputs, self.w_hid_out.T) + self.b_hid_reconstructed[-1]))
        for i in range(self.n_hid_layers - 1, 0, -1):
            self._hidden_reconstructed.insert(
                0, T.nnet.sigmoid(
                    T.dot(self._hidden_reconstructed[0], self.w_hid_hid[i-1].T) + self.b_hid_reconstructed[i-1]))
        self._inputs_reconstructed = T.nnet.sigmoid(
            T.dot(self._hidden_reconstructed[0], self.w_in_hid.T) + self.b_in)
        self.inputs_reconstructed = theano.function(
            inputs=[self.inputs], outputs=self._inputs_reconstructed)
        self._cost = T.mean(T.sum((self.inputs - self._inputs_reconstructed)**2, axis=1))
        self._grads = T.grad(self._cost, self.params)
        self.cost = theano.function(inputs=[self.inputs], outputs=self._cost)

if __name__ == '__main__':
    import cPickle
    import gzip
    import graddescent
    import matplotlib.pyplot as plt
    import utils
    f = gzip.open('/home/vincent/data/mnist.pkl.gz', 'rb')
    trainset, validset, testset = cPickle.load(f)
    f.close()

    ae = DenoisingAutoencoder(784, (10, 50, 10), 20)
    print 'instantiating trainer...'
    trainer = graddescent.GraddescentMinibatch_cliphidfilters_unloaded(
        ae, trainset[0], 100, 0.0001, momentum=0.9, loadsize=100)
    print 'starting training...'
    for i in range(10):
        trainer.step()
    plt.figure()
    utils.dispims(ae.w_in_hid.get_value(), 28, 28)
    plt.title('in->hid0')
    plt.savefig('/home/vincent/tmp/test_ae.png')

# vim: set ts=4 sw=4 sts=4 expandtab:
