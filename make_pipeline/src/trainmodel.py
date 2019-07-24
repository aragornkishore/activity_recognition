#!/usr/bin/env python
#-*- coding: utf-8 -*-

import optparse
import os

import matplotlib.pyplot as plt
import tables

import dictstruct
import disptools
import modelloader
import trainerloader


def parse_args():
    """Parses command-line arguments
    Returns:
        output of optparse.OptionParser.parse_args()
    """
    p = optparse.OptionParser()
    # add file options
    p.add_option(
        '-c', '--cfg', action='store', type='string', dest='cfg',
        help='path to config file (required)',
        metavar='FILE')

    p.add_option(
        '-w', '--sampleswhitefile', action='store', type='string',
        dest='sampleswhitefile',
        help='''h5 file in which the whitened samples are stored (required)''',
        metavar='FILE')

    p.add_option(
        '-m', '--modelfile', action='store', type='string',
        dest='modelfile',
        help='''h5 file in which the model params are stored (CONTENTS WILL
        BE OVERWRITTEN) (required)''',
        metavar='FILE')

    p.add_option(
        '-t', '--trainerfile', action='store', type='string',
        dest='trainerfile',
        help='''h5 file in which the trainer params are stored (CONTENTS WILL
        BE OVERWRITTEN) (required)''',
        metavar='FILE')

    p.add_option(
        '-p', '--tmpdir', action='store', type='string',
        dest='tmpdir',
        help='''directory in which temporary data (i.e. visualizations)
        should be stored''',
        metavar='DIR')

    (opts, _) = p.parse_args()
    # check if the required options are specified
    for opt in (opts.cfg, opts.sampleswhitefile,
                opts.modelfile, opts.trainerfile, opts.tmpdir):
        if opt is None:
            p.error('required parameter not provided')

    # check if the specified files/dirs really exist
    if not os.path.isfile(opts.cfg):
        p.error('config file {0} does not exist'.format(opts.cfg))
    if not os.path.isfile(opts.sampleswhitefile):
        p.error('whitened samples file {0} does not exist'.format(
            opts.sampleswhitefile))
    if not os.path.isdir(opts.tmpdir):
        p.error('tmp dir {0} does not exist'.format(opts.tmpdir))
    return opts

if __name__ == '__main__':
    options  = parse_args()
    cfg = dictstruct.DictStruct(options.cfg)

    # ensure number of input units is set
    datawhitefile = tables.openFile(options.sampleswhitefile, 'r')
    numvis = int(datawhitefile.root.inputs_white.shape[-1])
    W = datawhitefile.root.pca_params.W.read()
    test_inputs = datawhitefile.root.inputs_white.read(0, 10000)
    datawhitefile.close()

    model = modelloader.instantiate_model(cfg, numvis)
    trainer = trainerloader.instantiate_trainer(cfg, model, options.sampleswhitefile)
    for epoch in range(cfg.trainer.numepochs):
        trainer.step()
        if not ((epoch + 1) % 10):
            model.save(options.modelfile)
            trainer.save_params(options.trainerfile)
        if not ((epoch + 1) % 2):
            filters = model.Wl.get_value()
            if cfg.model.filter_disp_method == 'filter_movie' and \
               cfg.pca.method == 'blockwise':
                disptools.dispimsmovie(
                    os.path.join(
                        options.tmpdir,
                        'filters_{0:04d}epochs'.format(epoch + 1)),
                    W[:,:cfg.pca.nprinc],
                    filters,
                    cfg.data.framesize,
                    cfg.data.horizon)
                #TODO: generate histogram
                plt.figure()
                plt.clf()
                plt.hist(model.mappingsNonoise_batchwise(
                    test_inputs, batchsize=1000).flatten(), bins=100)
                plt.savefig(
                    os.path.join(
                        options.tmpdir,
                        'histogram_{0:04d}.png'.format(epoch + 1)))


    del trainer
    #datawhitefile.close()

# vim: set ts=4 sw=4 sts=4 expandtab:
