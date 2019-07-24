#!/usr/bin/env python
#-*- coding: utf-8 -*-

import os
import numpy as np
import tables
import theano
import theano.tensor as T
from fgae_sgdtrainer import FGAE_SGDTrainer

class FGAE_SGDPreTrainer_Momentum(FGAE_SGDTrainer):
    """This is a gradient descent trainer for the FGAE that uses momentum
    """

    def __init__(self, model, inputs, batchsize, loadsize, learningrate,
                 momentum=0.9, verbose=True, rng=None):
        """Basic setup for the SGD trainer with momentum
        Args:
            model: the model to be trained
            inputs: inputs of the training data
            batchsize: the size of a mini-batch
            loadsize: amount of data points copied in one go
            learningrate: the global learning rate
            momentum: amount of momentum to use [0..1]
            verbose: to print or not to print
            rng: numpy.random.RandomState object or None
        """
        # momentum: annealing factor of the running avg. gradients
        self.momentum = np.float32(momentum)
        self.learningrate = np.float32(learningrate)
        self.verbose = verbose
        self.batchsize = np.int32(batchsize)
        self.loadsize = np.int32(loadsize)
        if rng is None:
            self.rng = np.random.RandomState(1)
        else:
            self.rng = rng

        # dictionary of updates w/o
        self.updates_nomomentum = {}

        self.model = model

        # count batches, because we don't want to start using momentum right
        # from the start
        self.batchcount = np.int32(0)

        self.momentum_incs = \
          dict([(p, theano.shared(value=np.zeros(p.get_value().shape,
                            dtype=theano.config.floatX), name='momentum_inc_'+p.name))
                            for p in self.model.params_pretraining])

        self.momentum_inc_updates = {}
        self.momentum_updates = {}
        self.running_avg_grad_updates = {}

        self.costs_pretraining = []

        self.data_type = 'numpy'
        self.inputs = inputs

        self.epochcount = 0
        # create dict of param increments with the theano vars as keys
        self.incs = \
          dict([(p, theano.shared(value=np.zeros(p.get_value().shape,
                            dtype=theano.config.floatX), name='inc_'+p.name))
                            for p in self.model.params_pretraining])
        self.updates = {}
        self.inc_updates = {}
        self.nloads = self.inputs.shape[0] / loadsize
        if self.nloads == 0:
            raise ValueError('number of loads is zero')
        self.nbatches = loadsize / batchsize
        if self.nbatches == 0:
            raise ValueError('number of batches is zero')

        self.inputs_theano = theano.shared(
            self.inputs[:self.loadsize].reshape((-1, self.inputs.shape[-1])),
            name='inputs')

        self.compile_train_functions()


    def compile_train_functions(self):
        """
        Compiles the theano functions needed for updating the model
        """
        bidx = T.lscalar('batch_idx')

        for _param, _grad in zip(self.model.params_pretraining,
                                 self.model._grads_pretraining):
            # annealing of running avg. gradient
            self.momentum_inc_updates[self.momentum_incs[_param]] = \
                    self.momentum * self.incs[_param]

            # jump in direction of the running avg. grad
            self.momentum_updates[_param] = _param + self.momentum_incs[_param]
            # compute negative gradient and multiply with learning rate (&modifiers)
            try:
                self.inc_updates[self.incs[_param]] = - \
                        self.learningrate * \
                        self.model.learningrate_modifiers[_param.name] * _grad
            except AttributeError:
                try:
                    self.inc_updates[self.incs[_param]] = - \
                            self.learningrate * \
                            self.model.layer.learningrate_modifiers[_param.name] * _grad
                except:
                    self.inc_updates[self.incs[_param]] = - \
                            self.learningrate * _grad

            # correction using the current gradient
            self.updates[_param] = _param + self.incs[_param]
            # update the running avg. gradient
            self.running_avg_grad_updates[self.incs[_param]] = \
                    self.incs[_param] + self.momentum_incs[_param]

        # compute momentum term
        self._update_momentum_incs = theano.function(
            inputs=[],
            outputs=[],
            updates=self.momentum_inc_updates)#, mode='DEBUG_MODE')
        # apply momentum term (jump in direction of running avg. grad)
        self._add_momentum = theano.function(
            inputs=[],
            outputs=[],
            updates=self.momentum_updates)#, mode='DEBUG_MODE')

        # this function computes the gradients and the cost after the momentum
        # jump
        self._updateincs = theano.function(
            inputs=[bidx],
            outputs=self.model._cost_pretraining,
            updates=self.inc_updates,
            givens={
                self.model.inputs_pretraining_tm1:
                self.inputs_theano[bidx*2*self.batchsize:(bidx+1)*2*self.batchsize:2],
                self.model.inputs_pretraining_t:
                self.inputs_theano[bidx*2*self.batchsize + 1:
                                   (bidx+1)*2*self.batchsize:2]
            })#, mode='DEBUG_MODE')

        # this function applies the update and computes the cost after the
        # update
        self._trainmodel = theano.function(
            inputs=[bidx],
            outputs=self.model._cost_pretraining,
            updates=self.updates,
            givens={
                self.model.inputs_pretraining_tm1:
                self.inputs_theano[bidx*2*self.batchsize:(bidx+1)*2*self.batchsize:2],
                self.model.inputs_pretraining_t:
                self.inputs_theano[bidx*2*self.batchsize + 1:
                                   (bidx+1)*2*self.batchsize:2]
                #self.model.inputs_pretraining_tm1:
                #    self.inputs_theano[0, bidx*self.batchsize:(bidx+1)*self.batchsize],
                #    self.model.inputs_pretraining_t:
                #    self.inputs_theano[1, bidx*self.batchsize:(bidx+1)*self.batchsize]
            })#, mode='DEBUG_MODE')

        # this func updates the running avg. grad.
        self._update_running_grad = theano.function(
            inputs=[],
            outputs=[],
            updates=self.running_avg_grad_updates)#, mode='DEBUG_MODE')

    def step(self):
        cost_pretraining = 0.0
        stepcount = 0.0

        for load_idx in range(self.nloads):
            indices = np.random.permutation(self.loadsize)
            # first axis is always time => transpose
            self.inputs_theano.set_value(
                self.inputs[
                    load_idx * self.loadsize:
                    (load_idx + 1) * self.loadsize][
                        indices, :(self.inputs.shape[1] // 2)*2].reshape((
                            -1, self.inputs.shape[-1])))

            for batch_idx in self.rng.permutation(self.nbatches):
                self.batchcount += 1
                stepcount += 1.0
                # compute updates
                # start using momentum at 10th batch
                if self.batchcount < 10:
                    self._updateincs(batch_idx)
                    cost_pretraining = (1.0-1.0/stepcount)*cost_pretraining + \
                        (1.0/stepcount)*self._trainmodel(batch_idx)
                else:
                    self.batchcount = 10
                    self._update_momentum_incs()
                    self._add_momentum()
                    self._updateincs(batch_idx)
                    cost_pretraining = (1.0-1.0/stepcount)*cost_pretraining + \
                        (1.0/stepcount)*self._trainmodel(batch_idx)
                    self._update_running_grad()
        self.epochcount += 1
        self.costs_pretraining.append(cost_pretraining)
        if self.verbose:
            print '> epoch %d cost (pretrain): %r' % (
                self.epochcount, cost_pretraining)
        if np.isnan(cost_pretraining):
            raise ValueError, 'Cost function returned nan!'
        elif np.isinf(cost_pretraining):
            raise ValueError, 'Cost function returned infinity!'

    def load_params(self, filename):
        """Loads a pickled dictionary containing the trainers parameters from a file
        Params:
            filename: path to the file
        """
        ext = os.path.splitext(filename)[1]

        if ext == '.h5':
            print 'loading trainer params from a hdf5 file'
            self.load_h5(filename)

    def load_h5(self, filename):
        paramfile = tables.openFile(filename, 'r')
        self.learningrate = np.float32(paramfile.root.learningrate.read())
        self.verbose = paramfile.root.verbose.read()
        self.loadsize = int(paramfile.root.loadsize.read())
        self.batchsize = int(paramfile.root.batchsize.read())
        self.epochcount = int(paramfile.root.epochcount.read())
        self.costs_pretraining = paramfile.root.costs_pretraining.read()
        paramfile.close()
        self.compile_train_functions()


# vim: set ts=4 sw=4 sts=4 expandtab:
