#!/usr/bin/env python
#-*- coding: utf-8 -*-

import cPickle as pickle
import os
import shutil

import numpy as np
import tables
import theano
import theano.tensor as T


class FGAE_SGDTrainer(object):
    """Basic stochastic gradient descent (SGD) trainer
    """

    def __init__(self, model, input1, batchsize, loadsize, learningrate,
                 input2=None, verbose=True, rng=None):
        """Basic setup for the SGD trainer
        Args:
            model: the model to be trained
            input1: input1 of the training data
            batchsize: the size of a mini-batch
            loadsize: amount of data points copied in one go
            learningrate: the global learning rate
            input2: input2 of the training data (=None means equal to input1)
            verbose: to print or not to print
            rng: numpy.random.RandomState object or None
        """
        # copy the arguments to this object

        self.is_single_input = False
        if input2 is None:
            self.is_single_input = True
        if isinstance(input1, str):
            self.data_type = 'h5'
            self.input1file = tables.openFile(input1, 'r')
            self.input1 = self.input1file.root.inputs_white
        else:
            self.data_type = 'numpy'
            self.input1 = input1
        if not self.is_single_input:
            if self.data_type == 'h5':
                self.input2file = tables.openFile(input2, 'r')
                self.input2 = self.input2file.root.inputs_white
            else:
                self.input2 = input2
        self.model = model
        self.learningrate = np.float32(learningrate)
        self.verbose = verbose
        self.batchsize = np.int32(batchsize)
        self.loadsize = np.int32(loadsize)
        if rng is None:
            self.rng = np.random.RandomState(1)
        else:
            self.rng = rng

        if self.data_type == 'h5':
            self.input1_theano = theano.shared(
                self.input1.read(stop=self.loadsize),
                name='input1')
        else:
            self.input1_theano = theano.shared(self.input1[:self.loadsize],
                                               name='input1')

        if not self.is_single_input:
            if self.data_type == 'h5':
                self.input2_theano = theano.shared(
                    self.input2.read(stop=self.loadsize),
                    name='input2')
            else:
                self.input2_theano = theano.shared(
                    self.input2[:self.loadsize],
                    name='input2')

        self.epochcount = 0
        # create dict of param increments with the theano vars as keys
        self.incs = \
          dict([(p, theano.shared(value=np.zeros(p.get_value().shape,
                            dtype=theano.config.floatX), name='inc_'+p.name))
                            for p in self.model.params])
        self.updates = {}
        self.inc_updates = {}
        self.nloads = self.input1.shape[0] / loadsize
        if self.nloads == 0:
            raise ValueError('number of loads is zero')
        self.nbatches = loadsize / batchsize
        if self.nbatches == 0:
            raise ValueError('number of batches is zero')

        self.costs = []
        self.compile_train_functions()

    def __del__(self):
        if self.data_type == 'h5':
            self.input1file.close()
            if not self.is_single_input:
                self.input2file.close()

    def compile_train_functions(self):
        """
        Compiles the theano functions needed for updating the model
        """
        bidx = T.lscalar('batch_idx')

        for _param, _grad in zip(self.model.params, self.model._grads):
            self.inc_updates[self.incs[_param]] = - self.learningrate * _grad
            self.updates[_param] = _param + self.incs[_param]

        if self.is_single_input:
            # this function computes the gradients and the cost before the update
            self._updateincs = theano.function(
                inputs=[bidx],
                outputs=self.model._cost,
                updates=self.inc_updates,
                givens={self.model.inputs:
                        self.input1_theano[
                            bidx*self.batchsize:(bidx+1)*self.batchsize]
                })
        else:
            # this function computes the gradients and the cost before the update
            self._updateincs = theano.function(
                inputs=[bidx],
                outputs=self.model._cost,
                updates=self.inc_updates,
                givens={self.model.inputs:
                        self.input1_theano[
                            bidx*self.batchsize:(bidx+1)*self.batchsize],
                        self.model.inputs2:
                        self.input2_theano[
                            bidx*self.batchsize:(bidx+1)*self.batchsize]
                })

        if self.is_single_input:
            # this function applies the update and computes the cost after the
            # update
            self._trainmodel = theano.function(
                inputs=[bidx],
                outputs=self.model._cost,
                updates=self.updates,
                givens={self.model.inputs:
                        self.input1_theano[bidx*self.batchsize:(bidx+1)*self.batchsize]
                })
        else:
            # this function applies the update and computes the cost after the
            # update
            self._trainmodel = theano.function(
                inputs=[bidx],
                outputs=self.model._cost,
                updates=self.updates,
                givens={self.model.x:
                        self.input1_theano[bidx*self.batchsize:(bidx+1)*self.batchsize],
                        self.model.y:
                        self.input2_theano[bidx*self.batchsize:(bidx+1)*self.batchsize]
                })

    def step(self):
        """Performs one epoch of basic sgd training
        """
        cost = 0.0
        stepcount = 0.0

        for load_idx in range(self.nloads):
            indices = np.random.permutation(self.loadsize)
            if self.data_type == 'h5':
                self.input1_theano.set_value(
                    self.input1.read(start=load_idx * self.loadsize,
                                     stop=(load_idx + 1) * self.loadsize)[indices])
                if self.is_single_input:
                    self.input2_theano.set_value(
                        self.input2.read(start=load_idx * self.loadsize,
                                         stop=(load_idx + 1) * self.loadsize)[indices])
            else:
                self.input1_theano.set_value(
                    self.input1[load_idx * self.loadsize:
                                (load_idx + 1) * self.loadsize][indices])
                if self.is_single_input:
                    self.input2_theano.set_value(
                        self.input2[load_idx * self.loadsize:
                                    (load_idx + 1) * self.loadsize][indices])
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

    def save_params(self, filename):
        """Saves the trainers parameters to a file
        Params:
            filename: path to the file
        """
        ext = os.path.splitext(filename)[1]
        if ext == '.pkl':
            print 'saving trainer params to a pkl file'
            self.save_pkl(filename)
        else:
            print 'saving trainer params to a hdf5 file'
            self.save_h5(filename)

    def save_h5(self, filename):
        """Saves a HDF5 file containing the trainers parameters
        Params:
            filename: path to the file
        """
        try:
            shutil.copyfile(filename, '{}_bak'.format(filename))
        except IOError:
            print 'could not make backup of trainer param file (which is normal if we haven\'t saved one until now)'
        paramfile = tables.openFile(filename, 'w')
        paramfile.createArray(paramfile.root, 'learningrate', self.learningrate)
        paramfile.createArray(paramfile.root, 'verbose', self.verbose)
        paramfile.createArray(paramfile.root, 'loadsize', self.loadsize)
        paramfile.createArray(paramfile.root, 'batchsize', self.batchsize)
        paramfile.createArray(paramfile.root, 'epochcount', self.epochcount)
        paramfile.createArray(paramfile.root, 'costs', self.costs)
        paramfile.close()

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
        self.learningrate = paramfile.root.learningrate.read()
        self.verbose = paramfile.root.verbose.read()
        self.loadsize = paramfile.root.loadsize.read()
        self.batchsize = paramfile.root.batchsize.read()
        self.epochcount = paramfile.root.epochcount.read()
        self.costs = paramfile.root.costs.read()
        paramfile.close()
        self.compile_train_functions()

    def save_pkl(self, filename):
        """Saves a pickled dictionary containing the trainers parameters to a file
        Params:
            filename: path to the file
        """
        param_dict = {}
        param_dict['learningrate'] = self.learningrate
        param_dict['verbose'] = self.verbose
        param_dict['loadsize'] = self.loadsize
        param_dict['batchsize'] = self.batchsize
        param_dict['epochcount'] = self.epochcount
        param_dict['costs'] = self.costs
        pickle.dump(param_dict, open(filename, 'wb'))

# vim: set ts=4 sw=4 sts=4 expandtab:
