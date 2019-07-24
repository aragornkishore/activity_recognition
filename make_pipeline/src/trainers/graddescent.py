import os
import numpy as np
import shutil

import theano
import theano.tensor as T
import cPickle as pickle
import tables

HOME = os.environ['HOME']

class GraddescentMinibatch_cliphidfilters_unloaded(object):
    def __init__(self, model, inputs, batchsize, learningrate,
                 outputs_np = None, momentum=0.9, loadsize=None,
                 rng=None, verbose=True, valid_inputs=None,
                 valid_outputs=None, numcases=None):

        if isinstance(inputs, str):
            self.inputs_type = 'h5'
            self.h5file = tables.openFile(inputs, 'r')
            self.inputs = self.h5file.root.inputs_white
        else:
            self.inputs_type = 'numpy'
            self.inputs = inputs
        self.is_supervised_trainer = False
        if outputs_np is not None:
            self.is_supervised_trainer = True
        self.model         = model
        if self.is_supervised_trainer:
            self.outputs_np = outputs_np
        self.learningrate  = learningrate
        self.verbose       = verbose
        self.batchsize     = batchsize
        self.loadsize = loadsize
        if numcases is None or numcases > self.inputs.shape[0]:
            numcases = self.inputs.shape[0]
        self.numcases = numcases
        if self.batchsize > self.numcases:
            self.batchsize = self.numcases
        if self.loadsize == None:
            self.loadsize = self.batchsize * 100
        if self.loadsize > self.numcases:
            self.loadsize = self.numcases
        self.numloads      = self.numcases / self.loadsize

        if self.inputs_type == 'h5':
            self.inputs_theano = theano.shared(self.inputs.read(stop=self.loadsize))
        else:
            self.inputs_theano = theano.shared(self.inputs[:self.loadsize])
        self.use_validation = False
        if self.is_supervised_trainer:
            self.outputs = theano.shared(self.outputs_np[:self.loadsize].flatten().astype(np.int64))
            if valid_inputs != None:
                self.use_validation = True
                self.valid_inputs = valid_inputs
                self.valid_outputs = valid_outputs
                self.best_valid_loss = np.infty
                self.best_valid_loss_params = []
                for param_idx in range(len(model.params)):
                    self.best_valid_loss_params.append(model.params[param_idx].get_value())

        self.numbatches    = self.loadsize / batchsize
        self.momentum      = momentum
        self.momentum_batchcounter = 0
        if rng is None:
            self.rng = np.random.RandomState(1)
        else:
            self.rng = rng

        self.epochcount = 0
        self.index = T.lscalar()
        self.incs = \
          dict([(p, theano.shared(value=np.zeros(p.get_value().shape,
                            dtype=theano.config.floatX), name='inc_'+p.name))
                            for p in self.model.params])
        self.inc_updates = {}
        self.updates_nomomentum = {}
        self.updates = {}
        self.n = T.scalar('n')
        self.noop = 0.0 * self.n
        self.batch_idx = theano.shared(value=np.array(0, dtype=np.int64), name='batch_idx')
        self.set_learningrate(self.learningrate)


        #FIXME: DEBUG OUTPUT
        self.debug_f = theano.function(inputs=[], outputs=self.model.inputs, givens={
            self.model.inputs: self.inputs_theano[self.batch_idx*self.batchsize:(self.batch_idx+1)*self.batchsize],
        #self.model.outputs: self.outputs[self.batch_idx*self.batchsize:(self.batch_idx+1)*self.batchsize],
        },
        #mode='DEBUG_MODE'
        )
        #theano.printing.pydotprint(self.debug_f, '/tmp/test.png')

    def __del__(self):
        if self.inputs_type == 'h5':
            self.h5file.close()

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
        paramfile.createArray(paramfile.root, 'momentum', self.momentum)
        paramfile.createArray(paramfile.root, 'epochcount', self.epochcount)
        paramfile.createArray(paramfile.root, 'momentum_batchcounter', self.momentum_batchcounter)
        incsgrp = paramfile.createGroup(paramfile.root, 'incs', 'increments')
        for p in self.model.params:
            paramfile.createArray(incsgrp, p.name, self.incs[p].get_value())
        paramfile.close()

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
        param_dict['momentum'] = self.momentum
        param_dict['epochcount'] = self.epochcount
        param_dict['momentum_batchcounter'] = self.momentum_batchcounter
        param_dict['incs'] = dict([(p.name, self.incs[p].get_value()) for p in self.model.params])
        pickle.dump(param_dict, open(filename, 'wb'))

    def load_params(self, filename):
        """Loads a pickled dictionary containing the trainers parameters from a file
        Params:
            filename: path to the file
        """
        param_dict = pickle.load(open('%s' % filename, 'rb'))
        self.learningrate = param_dict['learningrate']
        self.verbose = param_dict['verbose']
        self.loadsize = param_dict['loadsize']
        self.batchsize = param_dict['batchsize']
        self.momentum = param_dict['momentum']
        self.epochcount = param_dict['epochcount']
        self.momentum_batchcounter = param_dict['momentum_batchcounter']
        for param_name in param_dict['incs'].keys():
            for p in self.model.params:
                if p.name == param_name:
                    self.incs[p].set_value(param_dict['incs'][param_name])
        self.numbatches = self.loadsize / self.batchsize
        self.numloads = self.inputs.shape[0] / self.loadsize
        if self.inputs_type == 'h5':
            self.inputs_theano.set_value(self.inputs.read(stop=self.loadsize))
        else:
            self.inputs_theano.set_value(self.inputs[:self.loadsize])
        if self.is_supervised_trainer:
            print "setting outputs"
            self.outputs.set_value(self.outputs_np[:self.loadsize])
        self.set_learningrate(self.learningrate)

    def set_learningrate(self, learningrate):
        self.learningrate = learningrate
        for _param, _grad in zip(self.model.params, self.model._grads):
            try:
                # Cliphid version:
                self.inc_updates[self.incs[_param]] = self.momentum * self.incs[_param] - self.learningrate * self.model.layer.learningrate_modifiers[_param.name] * _grad
                self.updates[_param] = _param + self.incs[_param]
                self.updates_nomomentum[_param] = _param - self.learningrate * self.model.layer.learningrate_modifiers[_param.name] * _grad
            except AttributeError:
                self.inc_updates[self.incs[_param]] = self.momentum * self.incs[_param] - self.learningrate * _grad
                self.updates[_param] = _param + self.incs[_param]
                self.updates_nomomentum[_param] = _param - self.learningrate * _grad

        if self.is_supervised_trainer:
            self._updateincs = theano.function([], self.model._cost, updates = self.inc_updates, givens = {self.model.inputs: self.inputs_theano[self.batch_idx*self.batchsize:(self.batch_idx+1)*self.batchsize],
                                                                                                                    self.model.outputs: self.outputs[self.batch_idx*self.batchsize:(self.batch_idx+1)*self.batchsize]})
        else:
            self._updateincs = theano.function([], self.model._cost, updates = self.inc_updates, givens = {self.model.inputs:self.inputs_theano[self.batch_idx*self.batchsize:(self.batch_idx+1)*self.batchsize]})
        if self.is_supervised_trainer:
            self._trainmodel = theano.function([self.n], self.noop, updates = self.updates)
        else:
            self._trainmodel = theano.function([self.n], self.noop, updates = self.updates)
        if self.is_supervised_trainer:
            self._trainmodel_nomomentum = theano.function([self.n], self.noop, updates = self.updates_nomomentum, givens = {self.model.inputs:self.inputs_theano[self.batch_idx*self.batchsize:(self.batch_idx+1)*self.batchsize],
                                                                                                                                       self.model.outputs:self.outputs[self.batch_idx*self.batchsize:(self.batch_idx+1)*self.batchsize]})
        else:
            self._trainmodel_nomomentum = theano.function([self.n], self.noop, updates = self.updates_nomomentum, givens = {self.model.inputs:self.inputs_theano[self.batch_idx*self.batchsize:(self.batch_idx+1)*self.batchsize]})
        self.momentum_batchcounter = 0

    def trainsubstep(self, batchidx):
        self.batch_idx.set_value(batchidx)
        stepcost = self._updateincs()


        if self.momentum_batchcounter < 10:
            self.momentum_batchcounter += 1
            self._trainmodel_nomomentum(0)
        else:
            self.momentum_batchcounter = 10
            self._trainmodel(0)
        return stepcost

    def validsubstep(self):
        """performs validation
        Returns: 0-1 loss on the validation data
        """
        valid_loss = np.sum(
            self.model.classify(self.valid_inputs)!=self.valid_outputs).astype(np.float32) / \
                self.valid_inputs.shape[0]
        if valid_loss < self.best_valid_loss:
            if self.verbose:
                print "updating best valid performance, was %f, is now %f" % (self.best_valid_loss, valid_loss)
            self.best_valid_loss = valid_loss
            for param_idx in range(len(self.model.params)):
                self.best_valid_loss_params[param_idx] = self.model.params[param_idx].get_value()

    def step(self):
        cost = 0.0
        stepcount = 0.0

        self.epochcount += 1

        for load_index in range(self.numloads):
            indices = np.random.permutation(self.loadsize)
            if self.inputs_type == 'h5':
                self.inputs_theano.set_value(self.inputs.read(start=load_index * self.loadsize, stop=(load_index + 1) * self.loadsize)[indices])
            else:
                self.inputs_theano.set_value(self.inputs[indices])
            if self.is_supervised_trainer:
                self.outputs.set_value(self.outputs_np[indices].flatten())
            for batch_index in self.rng.permutation(self.numbatches):
                stepcount += 1.0
                cost = (1.0-1.0/stepcount)*cost + (1.0/stepcount)*self.trainsubstep(batch_index)
            if self.use_validation:
                self.validsubstep()

            if self.verbose:
                print '> epoch {:d}, load {:d}/{:d}, cost: {:f}'.format(self.epochcount, load_index + 1, self.numloads, cost)
            # NOTE: to handle nan's in the cost function, put a try except ValueError around trainer.step()
            if np.isnan(cost):
                raise ValueError, 'Cost function returned nan!'
            elif np.isinf(cost):
                raise ValueError, 'Cost function returned infinity!'


