#!/usr/bin/env python
#-*- coding: utf-8 -*-

import os
import numpy as np
import tables
import theano
import theano.tensor as T
from fgae_sgdtrainer import FGAE_SGDTrainer

class FGAE_SGDTrainer_Momentum(FGAE_SGDTrainer):
    """This is a gradient descent trainer for the FGAE that uses momentum
    """

    def __init__(self, model, input1, batchsize, loadsize, learningrate,
                 momentum=0.9, input2=None, verbose=True, rng=None):
        """Basic setup for the SGD trainer with momentum
        Args:
            model: the model to be trained
            input1: input1 of the training data
            batchsize: the size of a mini-batch
            loadsize: amount of data points copied in one go
            learningrate: the global learning rate
            momentum: amount of momentum to use [0..1]
            input2: input2 of the training data (=None means equal to input1)
            verbose: to print or not to print
            rng: numpy.random.RandomState object or None
        """
        # momentum: annealing factor of the running avg. gradients
        self.momentum = np.float32(momentum)
        # dictionary of updates w/o
        self.updates_nomomentum = {}

        self.model = model

        # count batches, because we don't want to start using momentum right
        # from the start
        self.batchcount = np.int32(0)

        self.momentum_incs = \
          dict([(p, theano.shared(value=np.zeros(p.get_value().shape,
                            dtype=theano.config.floatX), name='momentum_inc_'+p.name))
                            for p in self.model.params])

        self.momentum_inc_updates = {}
        self.momentum_updates = {}
        self.running_avg_grad_updates = {}

        self.costs = []

        # call to parent constructor which does the basic stuff
        FGAE_SGDTrainer.__init__(self, model=self.model, input1=input1,
                                 batchsize=batchsize, loadsize=loadsize,
                                 learningrate=learningrate, input2=input2,
                                 verbose=verbose, rng=rng)

    def compile_train_functions(self):
        """
        Compiles the theano functions needed for updating the model
        """
        bidx = T.lscalar('batch_idx')

        for _param, _grad in zip(self.model.params, self.model._grads):
            # annealing of running avg. gradient
            self.momentum_inc_updates[self.momentum_incs[_param]] = \
                    self.momentum * self.incs[_param]
            #self.debug = theano.function(inputs=[],
            #                             outputs=(
            #                                 self.momentum *
            #                                 self.incs[self.model.b_y]))
            #self.debug2 = theano.function(inputs=[],
            #                              outputs=(
            #                                  self.momentum_incs[self.model.b_y]))
            #print 'incs: {0}'.format(self.debug().shape)
            #print 'momentum_incs: {0}'.format(self.debug2().shape)

            # jump in direction of the running avg. grad
            self.momentum_updates[_param] = _param + self.momentum_incs[_param]
            # compute negative gradient and multiply with learning rate (&modifiers)
            try:
                self.inc_updates[self.incs[_param]] = - \
                        self.learningrate * \
                        self.model.learningrate_modifiers[_param.name] * _grad
            except AttributeError:
                self.inc_updates[self.incs[_param]] = - \
                        self.learningrate * \
                        self.model.layer.learningrate_modifiers[_param.name] * _grad

            # correction using the current gradient
            self.updates[_param] = _param + self.incs[_param]
            # update the running avg. gradient
            self.running_avg_grad_updates[self.incs[_param]] = \
                    self.incs[_param] + self.momentum_incs[_param]

        # compute momentum term
        self._update_momentum_incs = theano.function(
            inputs=[],
            outputs=[],
            updates=self.momentum_inc_updates)
        # apply momentum term (jump in direction of running avg. grad)
        self._add_momentum = theano.function(
            inputs=[],
            outputs=[],
            updates=self.momentum_updates)

        # this function computes the gradients and the cost after the momentum
        # jump
        if self.is_single_input:
            self._updateincs = theano.function(
                inputs=[bidx],
                outputs=self.model._cost,
                updates=self.inc_updates,
                givens={self.model.inputs:
                        self.input1_theano[bidx*self.batchsize:(bidx+1)*self.batchsize]
                })
        else:
            self._updateincs = theano.function(
                inputs=[bidx],
                outputs=self.model._cost,
                updates=self.inc_updates,
                givens={self.model.inputs:
                        self.input1_theano[bidx*self.batchsize:(bidx+1)*self.batchsize],
                        self.model.inputs2:
                        self.input2_theano[bidx*self.batchsize:(bidx+1)*self.batchsize]
                })

        # this function applies the update and computes the cost after the
        # update
        if self.is_single_input:
            self._trainmodel = theano.function(
                inputs=[bidx],
                outputs=self.model._cost,
                updates=self.updates,
                givens={self.model.inputs:
                        self.input1_theano[bidx*self.batchsize:(bidx+1)*self.batchsize]
                })
        else:
            self._trainmodel = theano.function(
                inputs=[bidx],
                outputs=self.model._cost,
                updates=self.updates,
                givens={self.model.inputs:
                        self.input1_theano[bidx*self.batchsize:(bidx+1)*self.batchsize],
                        self.model.inputs2:
                        self.input2_theano[bidx*self.batchsize:(bidx+1)*self.batchsize]
                })

        # this func updates the running avg. grad.
        self._update_running_grad = theano.function(
            inputs=[],
            outputs=[],
            updates=self.running_avg_grad_updates)

    def step(self):
        """Only increments the epoch counter, training has to be implemented
        """
        cost = 0.0
        stepcount = 0.0

        for load_idx in range(self.nloads):
            indices = np.random.permutation(self.loadsize)
            if self.data_type == 'h5':
                self.input1_theano.set_value(
                    self.input1.read(start=load_idx * self.loadsize,
                                     stop=(load_idx + 1) * self.loadsize)[indices])
                if not self.is_single_input:
                    self.input2_theano.set_value(
                        self.input2.read(start=load_idx * self.loadsize,
                                         stop=(load_idx + 1) * self.loadsize)[indices])
            else:
                self.input1_theano.set_value(
                    self.input1[load_idx * self.loadsize:
                                (load_idx + 1) * self.loadsize][indices])
                if not self.is_single_input:
                    self.input2_theano.set_value(
                        self.input2[load_idx * self.loadsize:
                                    (load_idx + 1) * self.loadsize][indices])

            for batch_idx in self.rng.permutation(self.nbatches):
                self.batchcount += 1
                stepcount += 1.0
                # compute updates
                # start using momentum at 10th batch
                if self.batchcount < 10:
                    self._updateincs(batch_idx)
                    cost = (1.0-1.0/stepcount)*cost + \
                        (1.0/stepcount)*self._trainmodel(batch_idx)
                else:
                    self.batchcount = 10
                    self._update_momentum_incs()
                    self._add_momentum()
                    self._updateincs(batch_idx)
                    cost = (1.0-1.0/stepcount)*cost + \
                        (1.0/stepcount)*self._trainmodel(batch_idx)
                    self._update_running_grad()
        self.epochcount += 1
        self.costs.append(cost)
        if self.verbose:
            print '> epoch %d cost: %r' % (self.epochcount, cost)
        if np.isnan(cost):
            raise ValueError, 'Cost function returned nan!'
        elif np.isinf(cost):
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
        self.costs = paramfile.root.costs.read()
        paramfile.close()
        self.compile_train_functions()


# vim: set ts=4 sw=4 sts=4 expandtab:
