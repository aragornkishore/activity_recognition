#!/usr/bin/env python
#-*- coding: utf-8 -*-


def get_calc_mappings_func(cfg):
    if cfg.model.modeltype == 'stackedMCE':
        from models.stackedMCE import calc_mappings
        return calc_mappings

def instantiate_model(cfg, numvis, numvis2=None):
    """Instantiates a model
    Args:
        cfg: an object, that contains config of the model
    Returns:
        The model instance
    """

    if cfg.model.modeltype == 'sae':
        import numpy as np
        import theano
        numpy_rng  = np.random.RandomState(1)
        theano_rng = theano.tensor.shared_randomstreams.RandomStreams(1)
        from models import exp1_encoder
        return exp1_encoder.FactoredGatedAutoencoder(numpy_rng=numpy_rng, theano_rng=theano_rng,
                                                        nvis=cfg.model.numvis, nmap=cfg.model.nummap,
                                                        nhid = cfg.model.numfac, vistype=cfg.model.output_type)
    else:
        print 'model requested not found!!!'

def load_model(cfg, modelfile, numvis, numvis2=None):
    """Instantiates a model and loads the parameters
    Args:
        modeltype: a string indicating which model to use
        cfg.model: an object, that contains information about the model
    Returns:
        The model instance
    """
    model = instantiate_model(cfg, numvis, numvis2)
    model.load(modelfile)
    return model


# vim: set ts=4 sw=4 sts=4 expandtab:
