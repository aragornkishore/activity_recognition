""" Stacked mean-covariance encoder"""

import os
import shutil
import warnings

import numpy as np
import tables

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from logistic_sgd import LogisticRegression

SMALL = 0.000001


class factoredMCELayer(object):
    def __init__(self, inputs, numvis, numfac, nummap, numhid, output_type,
                 corruption_type='zeromask', corruption_level=0.0,
                 no_hid_filters=False, init_topology=None, weightcost=0.0,
                 halfparams=None, numpy_rng=None, theano_rng=None):
        self.numvis = numvis
        self.numfac = numfac
        self.nummap = nummap
        self.numhid = numhid
        self.no_hid_filters = no_hid_filters
        self.weightcost = weightcost
        self.output_type  = output_type
        self.use_mean_units = self.numhid > 0
        self.corruption_type = corruption_type
        self.corruption_level = corruption_level
        self.learningrate_modifiers = {}
        self.learningrate_modifiers['wyf']    = 1.0
        self.learningrate_modifiers['whid']   = 1.0
        self.learningrate_modifiers['wmf_in'] = 0.01
        self.learningrate_modifiers['wmf_out']    = 0.01
        self.learningrate_modifiers['bmap']   = 1.0
        self.learningrate_modifiers['bvis']   = 1.0
        self.learningrate_modifiers['bhid']   = 1.0

        self.meannyf = 0.01
        self.meannmf = 0.01

        if not numpy_rng:
            numpy_rng = np.random.RandomState(123)

        if not theano_rng:
            theano_rng = RandomStreams(123)

        self.inputs = inputs

        # SET UP VARIABLES AND PARAMETERS
        if init_topology is None:
            self.wmf_init = np.exp(numpy_rng.uniform(low=-3.0, high=-2.0, size=(nummap, numfac)).astype(theano.config.floatX))
        elif init_topology == '1d': #do topological initialization
            assert False
            assert np.mod(numfac, nummap) == 0
            stride = numfac / nummap
            self.wmf_init = np.zeros((nummap, numfac)).astype(theano.config.floatX)
            #ONE-D-TOPOLOGY:
            #first, we generate a simple stride-based, stretched-out 'eye'-matrix
            #then, we convolve (in horizontal direction only) to 'smear out' the 1's
            convkernel = np.ones(stride*2)
            #convkernel = np.exp(-np.linspace(-2.0, 2.0, stride*3)**2)
            convkernel /= convkernel.max()
            for i in range(nummap):
                self.wmf_init[i, i*stride] = 1
                self.wmf_init[i, :] = np.convolve(self.wmf_init[i, :], convkernel, mode='same')
            #--------------------------------------------------
        elif init_topology == '2d': #TWO-D-TOPOlOGY:
            import scipy.signal
            stride = numfac / nummap
            convkernel2d = np.ones((np.sqrt(stride)*2, np.sqrt(stride)*2))
            self.wmf_init = np.zeros((np.sqrt(nummap), np.sqrt(nummap), np.sqrt(numfac), np.sqrt(numfac)))
            for i in range(np.int(np.sqrt(nummap))):
                for j in range(np.int(np.sqrt(nummap))):
                    self.wmf_init[i, j, i*np.int(np.sqrt(stride)), j*np.int(np.sqrt(stride))] = 1
                    #self.wmf_init[i, j, :, :] = scipy.signal.convolve2d(self.wmf_init[i, j, :, :], convkernel2d, mode='same', old_behavior=False).astype('float32')
                    self.wmf_init[i, j, :, :] = scipy.signal.convolve2d(self.wmf_init[i, j, :, :], convkernel2d, mode='same').astype('float32')
                    #self.wmf_init[i, j, :, :] = scipy.signal.convolve2d(self.wmf_init[i, j, :, :], convkernel2d, mode='same').astype('float32')
            self.wmf_init = self.wmf_init.reshape(np.sqrt(nummap), np.sqrt(nummap), -1)
            self.wmf_init = self.wmf_init.transpose(2, 0, 1).reshape(-1, np.sqrt(nummap)**2).transpose(1, 0)
            self.wmf_init = self.wmf_init.astype(theano.config.floatX)
            #--------------------------------------------------

        self.topomask = (self.wmf_init > 0.0).astype(theano.config.floatX)

        self.wmf_out = theano.shared(value = 0.1*self.wmf_init, name='wmf_out')
        if halfparams is None:
            wyf_init = numpy_rng.uniform(low=-0.001, high=+0.001, size=(numvis, numfac)).astype(theano.config.floatX)
            self.wyf = theano.shared(value = wyf_init, name = 'wyf')
            self.wmf_in = theano.shared(value = 0.1*self.wmf_init, name='wmf_in')
            self.bmap = theano.shared(value = np.zeros(nummap, dtype=theano.config.floatX), name='bmap')
            whid_init = numpy_rng.uniform(low=-0.001, high=+0.001, size=(numvis, numhid)).astype(theano.config.floatX)
            self.whid = theano.shared(value = whid_init, name = 'whid')
            self.bhid = theano.shared(value = np.zeros(numhid, dtype=theano.config.floatX), name='bhid')
            self.bvis = theano.shared(value = np.zeros(numvis, dtype=theano.config.floatX), name='bvis')
        else:
            self.wyf = halfparams['wyf']
            self.wmf_in = halfparams['wmf_in']
            self.bmap = halfparams['bmap']
            self.bvis = theano.shared(value = np.zeros(numvis, dtype=theano.config.floatX), name='bvis')
            self.whid = halfparams['whid']
            self.bhid = halfparams['bhid']

        self.params = [self.wyf, self.bvis, self.bmap]
        if not self.no_hid_filters:
            self.params.extend([self.wmf_out, self.wmf_in])
        if self.use_mean_units:
            self.params.extend([self.whid, self.bhid])

        # DEFINE THE LAYER FUNCTION
        if self.corruption_type == 'zeromask':
            self._corruptedX = theano_rng.binomial(size=self.inputs.shape, n=1, p=1.0-self.corruption_level, dtype=theano.config.floatX) * self.inputs
            self._corruptedY = theano_rng.binomial(size=self.inputs.shape, n=1, p=1.0-self.corruption_level, dtype=theano.config.floatX) * self.inputs
        elif self.corruption_type == 'gaussian':
            self._corruptedX = theano_rng.normal(size=self.inputs.shape, avg=0.0, std=self.corruption_level, dtype=theano.config.floatX) + self.inputs
            self._corruptedY = theano_rng.normal(size=self.inputs.shape, avg=0.0, std=self.corruption_level, dtype=theano.config.floatX) + self.inputs
        elif self.corruption_type == 'xor':
            self._corruptedX = T.cast(T.xor(theano_rng.binomial(size=self.inputs.shape, n=1, p=self.corruption_level, dtype='int8'), T.cast(self.inputs, 'int8')), theano.config.floatX)
            self._corruptedY = T.cast(T.xor(theano_rng.binomial(size=self.inputs.shape, n=1, p=self.corruption_level, dtype='int8'), T.cast(self.inputs, 'int8')), theano.config.floatX)

        self._factorsX = T.dot(self._corruptedX, self.wyf)
        self._factorsY = T.dot(self._corruptedY, self.wyf)
        self._factorsNonoise = T.dot(self.inputs, self.wyf)
        if not self.no_hid_filters:
            self._mappings = T.nnet.sigmoid(T.dot(self._factorsX*self._factorsY, self.wmf_in.T)+self.bmap)
            self._mappingsNonoise = T.nnet.sigmoid(T.dot(self._factorsNonoise, self.wmf_in.T)+self.bmap)
            self._factorsH = T.dot(self._mappings, self.wmf_out)
        if self.no_hid_filters:
            self._mappings = T.nnet.sigmoid(self._factorsX*self._factorsY+self.bmap)
            self._mappingsNonoise = T.nnet.sigmoid(self._factorsNonoise+self.bmap)
            self._factorsH = self._mappings
        if self.use_mean_units:
            self._hiddensX = T.nnet.sigmoid(T.dot(self._corruptedX, self.whid)+self.bhid)
            self._hiddensY = T.nnet.sigmoid(T.dot(self._corruptedY, self.whid)+self.bhid)
            self._outputX_acts = T.dot(self._factorsY*self._factorsH, self.wyf.T)+self.bvis+T.dot(self._hiddensX, self.whid.T)
            self._outputY_acts = T.dot(self._factorsX*self._factorsH, self.wyf.T)+self.bvis+T.dot(self._hiddensY, self.whid.T)
        else:
            self._outputX_acts = T.dot(self._factorsY*self._factorsH, self.wyf.T)+self.bvis
            self._outputY_acts = T.dot(self._factorsX*self._factorsH, self.wyf.T)+self.bvis
        if self.output_type == 'binary':
            self._reconsX = T.nnet.sigmoid(self._outputX_acts)
            self._reconsY = T.nnet.sigmoid(self._outputY_acts)
        elif self.output_type == 'real':
            self._reconsX = self._outputX_acts
            self._reconsY = self._outputY_acts
        else:
            assert False, "unknown output type (has to be either 'binary' or 'real')"

        # ATTACH COST FUNCTIONS
        if self.output_type == 'binary':
            self._costpercase = - T.sum(
                 0.5*(self.inputs*T.log(self._reconsY) + (1-self.inputs)*T.log(1-self._reconsY))
                +0.5*(self.inputs*T.log(self._reconsX) + (1-self.inputs)*T.log(1-self._reconsX)),
                                   axis=1)
        elif self.output_type == 'real':
            self._costpercase = T.sum(0.5*((self.inputs-self._reconsX)**2)
                                     +0.5*((self.inputs-self._reconsY)**2), axis=1)
        else:
            assert False, "unknown output type (has to be either 'binary' or 'real')"

        self._cost = T.mean(self._costpercase) + self.weightcost * ((self.wyf**2).sum() + (self.wmf_in**2).sum() + (self.wmf_out**2).sum() + (self.whid**2).sum())
        self._grads = T.grad(self._cost, self.params)

    def normalizefilters(self):
        def inplacemult(x, v):
            x[:, :] *= v
            return x
        def inplacesubtract(x, v):
            x[:, :] -= v
            return x

        wyf = self.wyf.get_value(borrow=True)
        #wmf_out = self.wmf_out.get_value(borrow=True)

        nwyf = (wyf.std(0)+SMALL)[np.newaxis, :]
        #nwmf = (self.wmf_out.value.std(0)+SMALL)[np.newaxis, :]
        meannyf = nwyf.mean()
        #meannmf = nwmf.mean()
        self.meannyf = 0.95 * self.meannyf + 0.05 * meannyf
        #self.meannmf = 0.95 * self.meannmf + 0.05 * meannmf
        #print 'nmf: ', self.meannmf
        #print 'nyf: ', self.meannyf
        if self.meannmf > 1.5:
            self.meannmf = 1.5
        if self.meannyf > 1.5:
            self.meannyf = 1.5

        # CENTER FILTERS
        self.wyf.set_value(inplacesubtract(wyf, wyf.mean(0)[np.newaxis, :]), borrow=True)
        #self.wmf_out.set_value(inplacesubtract(wmf_out, wmf_out.mean(0)[np.newaxis,:]), borrow=True)

        # FIX STANDARD DEVIATION
        #self.wyf.set_value(inplacemult(wyf, 1.0/nwyf),borrow=True)
        #self.wmf_out.set_value(inplacemult(wmf_out, 1.0/nwmf), borrow=True)
        self.wyf.set_value(inplacemult(wyf, self.meannyf/nwyf), borrow=True)
        #self.wmf_out.set_value(inplacemult(wmf_out, self.meannmf/nwmf),borrow=True)

    def get_updates(self, learning_rate):
        updates = dict([(p, p - self.learningrate_modifiers[p.name] * learning_rate*g) for p, g in zip(self.params, self._grads)])
        return updates


class factoredMCEHalfLayer(object):
    """ This class defines a 'half' factored MCE layer for use in a stacked (unrolled) predictor. """

    def __init__(self, inputs, numvis, numfac, nummap, numhid, output_type,
                 corruption_type='zeromask', corruption_level=0.0,
                 no_hid_filters=False, init_topology=None,
                 numpy_rng=None, theano_rng=None):
        self.numvis  = numvis
        self.numfac  = numfac
        self.nummap  = nummap
        self.numhid  = numhid
        self.output_type  = output_type
        self.use_mean_units = self.numhid > 0
        self.corruption_type = corruption_type
        self.corruption_level = corruption_level
        self.no_hid_filters = no_hid_filters

        if not numpy_rng:
            numpy_rng = np.random.RandomState(1)

        if not theano_rng:
            theano_rng = RandomStreams(1)

        self.inputs = inputs

        # SET UP VARIABLES AND PARAMETERS
        wyf_init = numpy_rng.uniform(low=-0.01, high=+0.01, size=(numvis, numfac)).astype(theano.config.floatX)
        self.wyf = theano.shared(value = wyf_init, name = 'wyf')
        wmf_init = numpy_rng.uniform(low=-0.01, high=+0.01, size=(nummap, numfac)).astype(theano.config.floatX)
        #self.wmf_out = theano.shared(value = wmf_init, name = 'wmf_out')
        self.wmf_in = theano.shared(value = wmf_init, name = 'wmf_in')
        self.bmap  = theano.shared(value = np.zeros(nummap, dtype=theano.config.floatX), name='bmap')
        whid_init = numpy_rng.uniform(low=-0.01, high=+0.01, size=(numvis, numhid)).astype(theano.config.floatX)
        self.whid = theano.shared(value = whid_init, name = 'whid')
        self.bhid = theano.shared(value = np.zeros(numhid, dtype=theano.config.floatX), name='bhid')

        self.params = [self.wyf, self.bmap]
        if not self.no_hid_filters:
            self.params.extend([self.wmf_in])
        if self.use_mean_units:
            self.params.extend([self.whid, self.bhid])

        # DEFINE THE LAYER FUNCTION
        if self.corruption_type == 'zeromask':
            self._corruptedX = theano_rng.binomial(size=self.inputs.shape, n=1, p=1.0-self.corruption_level, dtype=theano.config.floatX) * self.inputs
            self._corruptedY = theano_rng.binomial(size=self.inputs.shape, n=1, p=1.0-self.corruption_level, dtype=theano.config.floatX) * self.inputs
        elif self.corruption_type == 'gaussian':
            self._corruptedX = theano_rng.normal(size=self.inputs.shape, avg=0.0, std=self.corruption_level, dtype=theano.config.floatX) + self.inputs
            self._corruptedY = theano_rng.normal(size=self.inputs.shape, avg=0.0, std=self.corruption_level, dtype=theano.config.floatX) + self.inputs

        self._factorsX = T.dot(self._corruptedX, self.wyf)
        self._factorsY = T.dot(self._corruptedY, self.wyf)
        self._factorsNonoise = T.dot(self.inputs, self.wyf)
        #self._factorsH = T.dot(self._mappings, self.wmf_out)
        if not self.no_hid_filters:
            self._mappings = T.nnet.sigmoid(T.dot(self._factorsX*self._factorsY, self.wmf_in.T)+self.bmap)
            self._mappingsNonoise = T.nnet.sigmoid(T.dot(self._factorsNonoise**2, self.wmf_in.T)+self.bmap)
        if self.no_hid_filters:
            self._mappings = T.nnet.sigmoid(self._factorsX*self._factorsY+self.bmap)
            self._mappingsNonoise = T.nnet.sigmoid(self._factorsNonoise**2+self.bmap)
        if self.use_mean_units:
            self._hiddensX = T.nnet.sigmoid(T.dot(self._corruptedX, self.whid)+self.bhid)
            self._hiddensY = T.nnet.sigmoid(T.dot(self._corruptedY, self.whid)+self.bhid)

        self.output = T.concatenate([self._mappings, self._hiddensX, self._hiddensY], 1)


class MCE(object):
    def __init__(self, numvis, numfac, nummap, numhid, output_type,
                       corruption_type='zeromask', corruption_level=0.2,
                       no_hid_filters=False, init_topology=None, weightcost=0.0,
                       numpy_rng=None, theano_rng=None):

        self.numvis = numvis
        self.numfac = numfac
        self.nummap = nummap
        self.numhid = numhid
        self.no_hid_filters = no_hid_filters
        self.weightcost = weightcost
        self.use_mean_units = self.numhid > 0
        self.output_type = output_type

        if not numpy_rng:
            numpy_rng = np.random.RandomState(1)

        if not theano_rng:
            theano_rng = RandomStreams(1)

        self.inputs = T.matrix('inputs')

        self.layer = factoredMCELayer(self.inputs, numvis, numfac, nummap, numhid, output_type,
                                     corruption_type=corruption_type, corruption_level=corruption_level,
                                     no_hid_filters=self.no_hid_filters, weightcost=self.weightcost,
                                     init_topology=init_topology,
                                     numpy_rng=numpy_rng, theano_rng=theano_rng)

        self.params = self.layer.params

        self._reconsX = self.layer._reconsX
        self._reconsY = self.layer._reconsY

        self._cost = self.layer._cost
        self._grads = self.layer._grads

        # COMPILE SOME FUNCTIONS THAT MAY BE USEFUL LATER
        self.corruptedX = theano.function([self.inputs], self.layer._corruptedX)
        self.corruptedY = theano.function([self.inputs], self.layer._corruptedY)
        self.mappingsNonoise = theano.function([self.inputs], self.layer._mappingsNonoise)
        self.mappings = theano.function([self.inputs], self.layer._mappings)
        if self.use_mean_units:
            self.hiddensX = theano.function([self.inputs], self.layer._hiddensX)
            self.hiddensY = theano.function([self.inputs], self.layer._hiddensY)
        self.reconsX = theano.function([self.inputs], self._reconsX)
        self.reconsY = theano.function([self.inputs], self._reconsY)
        self.cost = theano.function([self.inputs], self._cost)
        self.grads = theano.function([self.inputs], self._grads)
        self.grad = lambda inputs: np.concatenate([np.array(a).flatten() for a in self.grads(inputs)])

    def mappingsNonoise_batchwise(self, inputs, batchsize):
        numbatches = inputs.shape[0] / batchsize
        mappings = np.zeros((inputs.shape[0], self.nummap), dtype=theano.config.floatX)
        for batch in range(numbatches):
            mappings[batch*batchsize:(batch+1)*batchsize, :]=self.mappingsNonoise(inputs[batch*batchsize:(batch+1)*batchsize])
        if np.mod(inputs.shape[0], batchsize) > 0:
            mappings[numbatches*batchsize:, :] = self.mappingsNonoise(inputs[numbatches*batchsize:])
        return mappings

    #def __getstate__(self):
    #    return [p.value for p in self.params]

    #def __setstate__(self, d):
    #    print 'state not correct'
    #    for i in range(6):
    #        self.params[i].set_value(d[i])

    #def save(self, filename):
    #    cPickle.dump(dict([(p.name, p.value) for p in self.params]), open(filename, 'w'))

    def updateparams(self, newparams):
        def inplaceupdate(x, new):
            x[...] = new
            return x

        paramscounter = 0
        for p in self.params:
            pshape = p.get_value().shape
            pnum = np.prod(pshape)
            p.set_value(inplaceupdate(p.get_value(borrow=True), newparams[paramscounter:paramscounter+pnum].reshape(*pshape)), borrow=True)
            paramscounter += pnum

    def updateparams_fromdict(self, newparams):
        for p in self.params:
            p.set_value(newparams[p.name])

    def get_params_dict(self):
        return dict([(p.name, p.get_value()) for p in self.params])

    def get_params(self):
        return np.concatenate([p.get_value().flatten() for p in self.params])

    def save(self, filename):
        ext = os.path.splitext(filename)[1]
        if ext == '.h5':
            print 'saving h5 file'
            self.save_h5(filename)
        elif ext == '.npy':
            print 'saving npy file'
            self.save_npy(filename)
        elif ext == '.npz':
            print 'saving npz file'
            self.save_npz(filename)
        else:
            print 'unknown file extension: {0}'.format(ext)

    def save_h5(self, filename):
        try:
            shutil.copyfile(filename, '{0}_bak'.format(filename))
        except IOError:
            print 'could not make backup of model param file (which is normal if we haven\'t saved one until now)'

        h5file = tables.openFile(filename, 'w')
        for p in self.params:
            h5file.createArray(h5file.root, p.name, p.get_value())
            h5file.flush()
        h5file.close()

    def save_npy(self, filename):
        np.save(filename, self.get_params())

    def save_npz(self, filename):
        np.savez(filename, **(self.get_params_dict()))

    def load_h5(self, filename):
        h5file = tables.openFile(filename, 'r')
        new_params = {}
        for p in h5file.listNodes(h5file.root):
            new_params[p.name] = p.read()
        self.updateparams_fromdict(new_params)
        h5file.close()


    def load(self, filename):
        ext = os.path.splitext(filename)[1]
        if ext == '.h5':
            self.load_h5(filename)
        else:
            try:
                new_params = np.load(filename)
            except IOError, e:
                warnings.warn('''Parameter file could not be loaded with np.load()!
                            Is the filename correct?\n %s''' % (e, ))
            if type(new_params) == np.ndarray:
                print "loading npy file"
                self.updateparams(new_params)
            elif type(new_params) == np.lib.npyio.NpzFile:
                print "loading npz file"
                self.updateparams_fromdict(new_params)
            else:
                warnings.warn('''Parameter file loaded, but variable type not
                            recognized. Need npz or ndarray.''', Warning)


class stackedMCE(object):
    def __init__(self, numvis, numfacs=(256, 256), nummaps=(256, 256), numhids=(100, 100), numout=10,
                       numpy_rng=None, theano_rng=None,
                       corruption_type='zeromask', corruption_levels=(0.2, 0.2)):

        self.predictor_layers = []
        self.mce_layers  = []
        self.params     = []
        self.numlayers  = len(numfacs)
        assert len(numfacs) == len(nummaps) and len(nummaps) == len(numhids)

        if not numpy_rng:
            numpy_rng = np.random.RandomState(1)

        if not theano_rng:
            theano_rng = RandomStreams(1)

        self.x  = T.matrix('x')
        self.y  = T.ivector('y')

        for i in range(self.numlayers):
            if i == 0:
                input_size = numvis
                layer_input = self.x
            else:
                input_size = nummaps[i-1] + 2 * numhids[i-1]
                layer_input = self.predictor_layers[-1].output

            predictor_layer = factoredMCEHalfLayer(layer_input, input_size,
                                 numfacs[i], nummaps[i], numhids[i], output_type='binary',
                                 corruption_type=corruption_type, corruption_level=0.0,
                                 no_hid_filters=False, init_topology=None,
                                 numpy_rng=numpy_rng, theano_rng=theano_rng)

            self.predictor_layers.append(predictor_layer)
            self.params.extend(predictor_layer.params)

            mce_layer = factoredMCELayer(layer_input, input_size, numfacs[i], nummaps[i], numhids[i],
                                         'binary', corruption_type, corruption_levels[i],
                                         halfparams = dict([(p.name, p) for p in predictor_layer.params]),
                                         numpy_rng=numpy_rng, theano_rng=theano_rng)
            self.mce_layers.append(mce_layer)

        self.logreg_layer = LogisticRegression(input = self.predictor_layers[-1].output,
                                               n_in = nummaps[-1]+2*numhids[-1],
                                               n_out = numout)

        self.params.extend(self.logreg_layer.params)
        self.finetunecost = self.logreg_layer.negative_log_likelihood(self.y)
        self.zeroone = self.logreg_layer.errors(self.y)

    def pretraining_functions(self, train_set_x, batch_size):
        # index to a [mini]batch
        index            = T.lscalar('index')   # index to a minibatch
        #corruption_level = T.scalar('corruption')    # amount of corruption to use
        learning_rate    = T.scalar('learning_rate')    # learning rate to use
        # number of batches
        #n_batches = train_set_x.get_value().shape[0] / batch_size
        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin+batch_size

        pretrain_fns = []
        for mce in self.mce_layers:
            # get the cost and the updates list
            updates = mce.get_updates(learning_rate)
            # compile the theano function
            fn = theano.function(inputs = [index, theano.Param(learning_rate, default=0.1)],
                                 outputs = mce._cost,
                                 updates = updates,
                                 givens  = {self.x: train_set_x[batch_begin:batch_end]})
            # append `fn` to the list of functions
            pretrain_fns.append(fn)

        return pretrain_fns

    def build_finetune_functions(self, datasets, batch_size, learning_rate):
        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x , test_set_y ) = datasets[2]

        # compute number of minibatches for training, validation and testing
        n_valid_batches = valid_set_x.get_value().shape[0] / batch_size
        n_test_batches  = test_set_x.get_value().shape[0]  / batch_size

        index   = T.lscalar('index')    # index to a [mini]batch

        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.finetunecost, self.params)

        # compute list of fine-tuning updates
        updates = {}
        for param, gparam in zip(self.params, gparams):
            updates[param] = param - gparam*learning_rate

        train_fn = theano.function(inputs = [index],
              outputs =   self.finetunecost,
              updates = updates,
              givens  = {
                self.x : train_set_x[index*batch_size:(index+1)*batch_size],
                self.y : train_set_y[index*batch_size:(index+1)*batch_size]})

        test_score_i = theano.function([index], self.zeroone,
                 givens = {
                   self.x: test_set_x[index*batch_size:(index+1)*batch_size],
                   self.y: test_set_y[index*batch_size:(index+1)*batch_size]})

        valid_score_i = theano.function([index], self.zeroone,
              givens = {
                 self.x: valid_set_x[index*batch_size:(index+1)*batch_size],
                 self.y: valid_set_y[index*batch_size:(index+1)*batch_size]})

        # Create a function that scans the entire validation set
        def valid_score():
            return [valid_score_i(i) for i in xrange(n_valid_batches)]

        # Create a function that scans the entire test set
        def test_score():
            return [test_score_i(i) for i in xrange(n_test_batches)]

        return train_fn, valid_score, test_score

    def updateparams(self, newparams):
        def inplaceupdate(x, new):
            x[...] = new
            return x

        paramscounter = 0
        for p in self.params:
            pshape = p.get_value().shape
            pnum = np.prod(pshape)
            p.set_value(inplaceupdate(p.get_value(borrow=True), newparams[paramscounter:paramscounter+pnum].reshape(*pshape)), borrow=True)
            paramscounter += pnum

    def get_params(self):
        return np.concatenate([p.get_value().flatten() for p in self.params])

    def save(self, filename):
        np.save(filename, self.get_params())

    def load(self, filename):
        self.updateparams(np.load(filename))

