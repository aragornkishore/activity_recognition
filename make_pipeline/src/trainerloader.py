#!/usr/bin/env python
#-*- coding: utf-8 -*-


def instantiate_trainer(cfg, model, inputsfile, inputs2file=None):
    """Instantiates a model
    Args:
        cfg: an object, that contains config of the model
    Returns:
        The model instance
    """
    if cfg.trainer.trainertype == 'graddescent':
        from trainers import graddescent
        return graddescent.GraddescentMinibatch_cliphidfilters_unloaded(
            model=model, inputs=inputsfile,
            batchsize=cfg.trainer.batchsize,
            learningrate=cfg.trainer.learningrate,
            momentum=cfg.trainer.momentum,
            loadsize=cfg.trainer.loadsize, rng=None,
            verbose=cfg.trainer.verbose,
            numcases=cfg.trainer.numcases)
    elif cfg.trainer.trainertype == 'fgae_sgd':
        from trainers import fgae_sgdtrainer
        if inputs2file is None:
            return fgae_sgdtrainer.FGAE_SGDTrainer(
                model=model, data=inputsfile, batchsize=cfg.trainer.batchsize,
                loadsize=cfg.trainer.loadsize, learningrate=cfg.trainer.learningrate,
                verbose=cfg.trainer.verbose, rng=None)
        else:
            return fgae_sgdtrainer.FGAE_SGDTrainer(
                model=model, input1=inputsfile, batchsize=cfg.trainer.batchsize,
                loadsize=cfg.trainer.loadsize, learningrate=cfg.trainer.learningrate,
                verbose=cfg.trainer.verbose, rng=None, input2=inputs2file)
    elif cfg.trainer.trainertype == 'fgae_sgd_momentum':
        from trainers import fgae_sgdtrainer_momentum
        if inputs2file is None:
            return fgae_sgdtrainer_momentum.FGAE_SGDTrainer_Momentum(
                model=model, input1=inputsfile, batchsize=cfg.trainer.batchsize,
                loadsize=cfg.trainer.loadsize, learningrate=cfg.trainer.learningrate,
                momentum=cfg.trainer.momentum,
                verbose=cfg.trainer.verbose, rng=None)
        else:
            return fgae_sgdtrainer_momentum.FGAE_SGDTrainer_Momentum(
                model=model, input1=inputsfile, batchsize=cfg.trainer.batchsize,
                loadsize=cfg.trainer.loadsize, learningrate=cfg.trainer.learningrate,
                momentum=cfg.trainer.momentum,
                verbose=cfg.trainer.verbose, rng=None, input2=inputs2file)

def load_trainer(cfg, model, trainerfile, inputsfile, inputs2file=None):
    """Instantiates a model and loads the parameters
    Args:
        modeltype: a string indicating which model to use
        cfg.model: an object, that contains information about the model
    Returns:
        The model instance
    """
    trainer = instantiate_trainer(cfg, model, inputsfile, inputs2file)
    trainer.load_params(trainerfile)
    return trainer

# vim: set ts=4 sw=4 sts=4 expandtab:
