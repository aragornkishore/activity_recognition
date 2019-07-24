#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np
import theano
import theano.tensor as T

# TODO: define test case and compare convergence/performance (e.g. natural images or
# shifts)
# TODO: first implement simple rmsprop, then with Nesterov-momentum

class SGDTrainer(object):
    """Basic stochastic gradient descent (SGD) trainer
    """

    def __init__(self, model, data, batchsize, loadsize, learningrate,
                 verbose=True, rng=None):
        """Basic setup for the trainer
        Args:
            model: the model to be trained
            data: the training data
            batchsize: the size of a mini-batch
            loadsize: amount of data points copied in one go
            learningrate: the global learning rate
            verbose: to print or not to print
            rng: numpy.random.RandomState object or None
        """
        # copy the arguments to this object
        self.model = model
        if type(data) == np.ndarray or np.core.memmap:
            self.data_np = data
        else:
            raise NotImplementedError('this trainer can only deal with ndarrays')
        self.learningrate = learningrate
        self.verbose = verbose
        self.batchsize = batchsize
        self.loadsize = loadsize
        if rng is None:
            self.rng = np.random.RandomState(1)
        else:
            self.rng = rng
        self.data = theano.shared(self.data_np[:self.loadsize], name='data')
        self.epochcount = 0
        # create dict of param increments with the theano vars as keys
        self.incs = \
          dict([(p, theano.shared(value=np.zeros(p.get_value().shape,
                            dtype=theano.config.floatX), name='inc_'+p.name))
                            for p in self.model.params])
        self.updates = {}
        self.inc_updates = {}
        self.nloads = data.shape[0] / loadsize
        self.nbatches = loadsize / batchsize

        self.costs = []
        self.compile_train_functions()

    def compile_train_functions(self):
        """
        Compiles the theano functions needed for updating the model
        """
        bidx = T.lscalar('batch_idx')

        for _param, _grad in zip(self.model.params, self.model._grads):
            self.inc_updates[self.incs[_param]] = - self.learningrate * _grad
            self.updates[_param] = _param + self.incs[_param]

        # this function computes the gradients and the cost before the update
        self._updateincs = theano.function(
            inputs=[bidx],
            outputs=self.model._cost,
            updates=self.inc_updates,
            givens={self.model.inputs:
                    self.data[bidx*self.batchsize:(bidx+1)*self.batchsize]
            })
        # this function applies the update and computes the cost after the
        # update
        self._trainmodel = theano.function(
            inputs=[bidx],
            outputs=self.model._cost,
            updates=self.updates,
            givens={self.model.inputs:
                    self.data[bidx*self.batchsize:(bidx+1)*self.batchsize]
            })

    def step(self):
        """Performs one epoch of basic sgd training
        """
        cost = 0.0
        stepcount = 0.0

        for load_idx in range(self.nloads):
            self.data.set_value(self.data_np[np.random.permutation(self.data_np.shape[0])[:self.loadsize]])
            for batch_idx in self.rng.permutation(self.nbatches):
                stepcount += 1.0
                self._updateincs(batch_idx)
                cost = (1.0-1.0/stepcount)*cost + \
                       (1.0/stepcount)*self._trainmodel(batch_idx)
        self.epochcount += 1
        self.costs.append(cost)
        if self.verbose:
            print '> epoch %d cost: %r' % (self.epochcount, cost)
        if np.isnan(cost):
            raise ValueError, 'Cost function returned nan!'
        elif np.isinf(cost):
            raise ValueError, 'Cost function returned infinity!'


class SGDTrainer_Momentum(SGDTrainer):
    """This is a gradient descent trainer that uses momentum
    """
    def __init__(self, model, data, batchsize, loadsize, learningrate,
                 momentum=0.9, verbose=True, rng=None):
        """Basic setup for the momentum SGD trainer
        The gradient is basically used to update velocity.
        Imagine a ball on the error surface, that first rolls in direction of
        steepest descent, but soon picks up a momentum. This should prevent
        oscillations orthogonal to elongated valleys in the error surface.

        Args:
            model: the model to be trained
            data: the training data
            batchsize: the size of a mini-batch
            loadsize: amount of data points copied in one go
            learningrate: the global learning rate
            momentum: decay factor for momentum
            verbose: to print or not to print
            rng: numpy.random.RandomState object or None
        """
        # momentum: annealing factor of the running avg. gradients
        self.momentum = momentum
        # dictionary of updates w/o
        self.updates_nomomentum = {}

        # count batches, because we don't want to start using momentum right
        # from the start
        self.batchcount = 0

        self.momentum_incs = \
          dict([(p, theano.shared(value=np.zeros(p.get_value().shape,
                            dtype=theano.config.floatX), name='momentum_inc_'+p.name))
                            for p in model.params])

        self.momentum_inc_updates = {}
        self.momentum_updates = {}
        self.running_avg_grad_updates = {}

        self.costs = []

        # call to parent constructor which does the basic stuff
        SGDTrainer.__init__(self, model, data, batchsize, loadsize, learningrate,
                            verbose, rng)

    def compile_train_functions(self):
        """
        Compiles the theano functions needed for updating the model
        """
        bidx = T.lscalar('batch_idx')

        for _param, _grad in zip(self.model.params, self.model._grads):
            # annealing of running avg. gradient
            self.momentum_inc_updates[self.momentum_incs[_param]] = self.momentum * self.incs[_param]
            # jump in direction of the running avg. grad
            self.momentum_updates[_param] = _param + self.momentum_incs[_param]
            # compute negative gradient and multiply with learning rate (&modifiers)
            self.inc_updates[self.incs[_param]] = - self.learningrate * _grad
            # correction using the current gradient
            self.updates[_param] = _param + self.incs[_param]
            # update the running avg. gradient
            self.running_avg_grad_updates[self.incs[_param]] = self.incs[_param] + self.momentum_incs[_param]

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
        self._updateincs = theano.function(
            inputs=[bidx],
            outputs=self.model._cost,
            updates=self.inc_updates,
            givens={self.model.inputs:
                    self.data[bidx*self.batchsize:(bidx+1)*self.batchsize]
            })
        # this function applies the update and computes the cost after the
        # update
        self._trainmodel = theano.function(
            inputs=[bidx],
            outputs=self.model._cost,
            updates=self.updates,
            givens={self.model.inputs:
                    self.data[bidx*self.batchsize:(bidx+1)*self.batchsize]
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
        self.data_np = self.data_np[np.random.permutation(self.data_np.shape[0])]

        for load_idx in range(self.nloads):
            self.data.set_value(self.data_np[load_idx * self.loadsize: (load_idx+1) * self.loadsize])
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

class SGDTrainer_AdaptiveLearningRate(SGDTrainer_Momentum):
    def __init__(self, model, data, batchsize, loadsize, learningrate,
                 gain_increment=0.05, gain_decay_factor=0.95,
                 gain_limits=(0.1,10), momentum=0.9,
                 verbose=True, rng=None):
        """Basic setup for the SGD trainer with adaptive learning rate
        The global learning rate is multiplied with local gain, that is
        increased additively when direction is stable (the weight gradients
        don't change signs), and decreased when oscillations occur.

        Args:
            model: the model to be trained
            data: the training data
            batchsize: the size of a mini-batch
            loadsize: amount of data points copied in one go
            learningrate: the global learning rate
            gain_increment: float that is added to l.r. if direction stable
            gain_decay_factor: float multiplied w/ l.r. if oscillations occur
            gain_limits: lower and upper limit for the gain
            momentum: decay factor for momentum
            verbose: to print or not to print
            rng: numpy.random.RandomState object or None
        """
        # copy the gain hyper parameters
        self.gain_increment = theano.shared(np.float32(gain_increment))
        self.gain_decay_factor = theano.shared(np.float32(gain_decay_factor))
        self.gain_min = theano.shared(np.float32(gain_limits[0]))
        self.gain_max = theano.shared(np.float32(gain_limits[1]))

        self.gains = \
          dict([(p, theano.shared(value=np.ones(p.get_value().shape,
                            dtype=theano.config.floatX), name='gain_'+p.name))
                            for p in model.params])

        # call to parent constructor which does the basic stuff
        SGDTrainer_Momentum.__init__(self, model, data, batchsize, loadsize, learningrate,
                            momentum, verbose, rng)


    def compile_train_functions(self):
        """
        Compiles the theano functions needed for updating the model
        """
        bidx = T.lscalar('batch_idx')

        for _param, _grad in zip(self.model.params, self.model._grads):
            # annealing of running avg. gradient
            self.momentum_inc_updates[self.momentum_incs[_param]] = \
                    self.momentum * self.incs[_param]
            # jump in direction of the running avg. grad
            self.momentum_updates[_param] = _param + self.momentum_incs[_param]
            # compute neg. gradient and multiply with learning rate (&modifiers)
            # in this trainer, we also update the gains
            self.inc_updates[self.incs[_param]] = - self.learningrate * \
                    self.gains[_param] * _grad
            # if accumulated gradient and current grad have the same sign
            # increase gain, else we decay the gain

            self.inc_updates[self.gains[_param]] = \
                T.clip(
                    T.switch(T.gt(self.incs[_param] * _grad, 0.0), # elementwise if else
                            self.gains[_param] + self.gain_increment,
                            self.gains[_param] * self.gain_decay_factor),
                    self.gain_min, self.gain_max)
            # correction using the current gradient
            self.updates[_param] = _param + self.incs[_param]
            # update the running avg. gradient
            self.running_avg_grad_updates[self.incs[_param]] = self.incs[_param] + self.momentum_incs[_param]

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
        self._updateincs = theano.function(
            inputs=[bidx],
            outputs=self.model._cost,
            updates=self.inc_updates,
            givens={self.model.inputs:
                    self.data[bidx*self.batchsize:(bidx+1)*self.batchsize]
            })
        # this function applies the update and computes the cost after the
        # update
        self._trainmodel = theano.function(
            inputs=[bidx],
            outputs=self.model._cost,
            updates=self.updates,
            givens={self.model.inputs:
                    self.data[bidx*self.batchsize:(bidx+1)*self.batchsize]
            })
        # this func updates the running avg. grad.
        self._update_running_grad = theano.function(
            inputs=[],
            outputs=[],
            updates=self.running_avg_grad_updates)



# vim: set ts=4 sw=4 sts=4 expandtab:
