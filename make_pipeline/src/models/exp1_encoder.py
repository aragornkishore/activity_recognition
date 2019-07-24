import numpy, pylab
import cPickle
import warnings

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from theano.tensor.nnet import conv


class FactoredGatedAutoencoder(object):

    def __init__(self, numpy_rng, theano_rng = None, W=None, bvis = None, bhid = None, nvis=169, nmap=256, nhid=128,
            vistype = 'binary', reg_strength=1.):
        self.nvis  = nvis
        self.nmap = nmap
        self.nhid = nhid
        self.vistype = vistype
        self.reg_strength = reg_strength       
 
        if not theano_rng : 
            theano_rng = RandomStreams(rng.randint(2**30))

        self.Wl = theano.shared(numpy.asarray(numpy_rng.uniform(
                        low=-0.01,
                        high=0.01,
                        size=(nvis,nmap)), dtype=theano.config.floatX),name ='Wl')

        self.bvisl = theano.shared(numpy.zeros((nvis,), 
                        dtype=theano.config.floatX), name='bvisl')

        self.bvisr = theano.shared(numpy.zeros((nvis,), 
                        dtype=theano.config.floatX), name='bvisr')


        self.inputs = T.matrix(name = 'inputs')
        self.theano_rng = theano_rng
        n = nvis
        self.input_l = self.inputs
        self.input_r = self.inputs
        self.params = [self.Wl, self.bvisl, self.bvisr]
 

        self.features_l,self.features_r = self.get_features(self.input_l,self.input_r)
        self.product = T.nnet.sigmoid(self.features_l*self.features_r)
         

        self.zl,self.zr   = self.get_reconstructed_input(self.product)

        if self.vistype == 'binary':
                Ll = - T.sum( self.input_l*T.log(self.zl) + (1-self.input_l)*T.log(1-self.zl), axis=1 )
                Lr = - T.sum( self.input_r*T.log(self.zr) + (1-self.input_r)*T.log(1-self.zr), axis=1 )
        elif self.vistype == 'real':
                Ll = T.sum(0.5 * ((self.input_l - self.zl)**2), axis=1)
                Lr = T.sum(0.5 * ((self.input_r - self.zr)**2), axis=1)

        contractive_cost_l = T.sum( ((self.product * (1 - self.product))**2) * T.sum(self.Wl**2, axis=0) * self.features_r**2, axis=1)
        Ll = Ll + self.reg_strength * contractive_cost_l

        contractive_cost_r = T.sum( ((self.product * (1 - self.product))**2) * T.sum(self.Wl**2, axis=0) * self.features_l**2, axis=1) 
        Lr = Lr + self.reg_strength * contractive_cost_r

        self._cost = T.mean(Ll) + T.mean(Lr) 
        self._grads = T.grad(self._cost, self.params)
        
        self.get_product = theano.function([self.inputs],self.product)

    def get_corrupted_input(self, input):

        if self.corruption_type == 'deleted':
            return  self.theano_rng.binomial( size = input.shape, n = 1, p =  1 - self.corruption_level, dtype=theano.config.floatX) * input
        elif self.corruption_type == 'gaussian':
            return  self.theano_rng.normal( size = input.shape, avg=0.0, std = self.corruption_level, dtype=theano.config.floatX) + input
    
    def get_features(self, yl, yr):
    
        out_l = T.dot(yl,self.Wl)
        out_r = T.dot(yr,self.Wl)

        return (out_l ,out_r)

    def get_hidden_values(self,product):
        hid = T.dot(product,self.Wh)    
        h = T.nnet.sigmoid(hid + self.bhidh)
        return (h)

    def get_reconstructed_input(self, hidden ):    
        recon_h = hidden
        out_l = T.dot(recon_h*self.features_r,self.Wl.T)
        if self.vistype == 'binary':
            recon_l = T.nnet.sigmoid(out_l + self.bvisl)
        elif self.vistype == 'real':
            recon_l = (out_l + self.bvisl)

        out_r = T.dot(recon_h*self.features_l,self.Wl.T)    
        if self.vistype == 'binary':
            recon_r = T.nnet.sigmoid(out_r + self.bvisr)
        elif self.vistype == 'real':
            recon_r = (out_r + self.bvisr)

        return recon_l,recon_r


    def normalizefilters(self):
        def inplacemult(x, v):
                x[:, :] *= v
                return x
        def inplacesubtract(x, v):
                x[:, :] -= v
                return x

        nl = (self.Wl.get_value().std(0)+0.001)[numpy.newaxis, :]
        Wl = self.Wl.get_value(borrow=True)
        self.Wl.set_value(inplacesubtract(Wl, Wl.mean(0)[numpy.newaxis, :]), borrow=True)
        self.Wl.set_value(inplacemult(Wl, nl.mean()/nl),borrow=True)

    def mappingsNonoise_batchwise(self, input, batchsize):
        numbatches = input.shape[0] / batchsize
        mappings = numpy.zeros((input.shape[0], self.nmap), dtype=theano.config.floatX)
        for batch in range(numbatches):
            mappings[batch*batchsize:(batch+1)*batchsize, :]=self.get_product(input[batch*batchsize:(batch+1)*batchsize])
        if numpy.mod(input.shape[0], batchsize) > 0:
            mappings[numbatches*batchsize:, :]=self.get_product(input[numbatches*batchsize:])
        return mappings


    def updateparams(self, newparams):
        def inplaceupdate(x, new):
            x[...] = new
            return x

        paramscounter = 0
        for p in self.params:
            pshape = p.get_value().shape
            pnum = numpy.prod(pshape)
            p.set_value(inplaceupdate(p.get_value(borrow=True), newparams[paramscounter:paramscounter+pnum].reshape(*pshape)), borrow=True)
            paramscounter += pnum

    def updateparams_fromdict(self, newparams):
        for p in self.params:
            p.set_value(newparams[p.name])

    def get_params_dict(self):
        return dict([(p.name, p.get_value()) for p in self.params])

    def get_params(self):
        return numpy.concatenate([p.get_value().flatten() for p in self.params])

    def save(self, filename):
        numpy.save(filename, self.get_params())

    def save_npz(self, filename):
        numpy.savez(filename, **(self.get_params_dict()))

    def load(self, filename):
        new_params = None
        try:
            new_params = numpy.load(filename)
        except IOError, e:
            warnings.warn('''Parameter file could not be loaded with numpy.load()!
                          Is the filename correct?\n %s''' % (e, ))
        if type(new_params) == numpy.ndarray:
            print "loading npy file"
            self.updateparams(new_params)
        elif type(new_params) == numpy.lib.npyio.NpzFile:
            print "loading npz file"
            self.updateparams_fromdict(new_params)
        else:
            warnings.warn('''Parameter file loaded, but variable type not
                          recognized. Need npz or ndarray.''', Warning)

