#!/usr/bin/env python
#-*- coding: utf-8 -*-

import optparse
import os

import numpy as np
import tables

import dictstruct
import pca


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
        '-i', '--samplesfile', action='store', type='string',
        dest='samplesfile',
        help='h5 file in which the samples are stored (required)',
        metavar='FILE')

    p.add_option(
        '-p', '--pcafile', action='store', type='string', dest='pcafile',
        help='''h5 file to store the pca params in
        (CONTENT WILL BE OVERWRITTEN) (required)''',
        metavar='FILE')

    (opts, _) = p.parse_args()
    # check if the required options are specified
    for opt in (opts.cfg, opts.samplesfile, opts.pcafile):
        if opt is None:
            p.error('required parameter not provided')


    # check if the specified files really exist
    if not os.path.isfile(opts.cfg):
        p.error('config file {0} does not exist'.format(opts.cfg))
    if not os.path.isfile(opts.samplesfile):
        p.error('samples file {0} does not exist'.format(opts.samplesfile))
    return opts


if __name__ == '__main__':
    options  = parse_args()
    cfg = dictstruct.DictStruct(options.cfg)

    datafile = tables.openFile(options.samplesfile, 'r')
    if cfg.pca.method == 'blockwise':
        # compute whitening params for blocks
        V, W, m0, s0, var_fracs = pca.pca(
            datafile.root.inputs.read(stop=cfg.pca.numsamples),
            use_gpu=cfg.pca.use_gpu,
            batchsize=cfg.pca.batchsize,
            verbose=cfg.pca.verbose)
    else:
        # compute whitening params for frames by reshaping the data matrix
        V, W, m0, s0, var_fracs = pca.pca(
            datafile.root.inputs.read(
                stop=cfg.pca.numsamples).reshape(
                    (-1, cfg.data.framesize * cfg.data.framesize)),
            use_gpu=cfg.pca.use_gpu,
            batchsize=cfg.pca.batchsize,
            verbose=cfg.pca.verbose)
    datafile.close()

    print V, W.shape, m0.shape, s0.shape
    pca_param_file = tables.openFile(options.pcafile, 'w')
    pcagrp = pca_param_file.createGroup(
        pca_param_file.root,
        'pca_params', 'PCA Whitening parameters')
    pca_param_file.createArray(pcagrp, 'V', V)
    pca_param_file.createArray(pcagrp, 'W', W)
    pca_param_file.createArray(pcagrp, 'm0', m0)
    pca_param_file.createArray(pcagrp, 's0', s0)
    pca_param_file.createArray(pcagrp, 'var_fracs', var_fracs)
    pca_param_file.createArray(pcagrp, 'method', cfg.pca.method)
    nprincomps = np.where(var_fracs > cfg.pca.retain_var)[0][0]
    retained_var = var_fracs[nprincomps]
    pca_param_file.createArray(pcagrp, 'nprincomps', nprincomps)
    pca_param_file.createArray(pcagrp, 'retained_var', retained_var)
    pca_param_file.close()

# vim: set ts=4 sw=4 sts=4 expandtab:
