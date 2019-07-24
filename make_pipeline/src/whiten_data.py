#!/usr/bin/env python
#-*- coding: utf-8 -*-

import optparse
import os
import shutil

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
    p.add_option( '-i', '--samplesfile', action='store', type='string',
        dest='samplesfile',
        help='h5 file in which the samples are stored (required)',
        metavar='FILE')

    p.add_option(
        '-p', '--pcafile', action='store', type='string', dest='pcafile',
        help='''h5 file to store the pca params in
        (CONTENT WILL BE OVERWRITTEN) (required)''',
        metavar='FILE')

    p.add_option(
        '-w', '--sampleswhitefile', action='store', type='string',
        dest='sampleswhitefile',
        help='''h5 file in which the whitened samples are stored (CONTENTS WILL
        BE OVERWRITTEN) (required)''',
        metavar='FILE')


    (opts, _) = p.parse_args()
    # check if the required options are specified
    for opt in (opts.cfg, opts.samplesfile, opts.pcafile,
                opts.sampleswhitefile):
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
    # copy preprocessing file to preprocessed_inputs file
    # because we want to store the params together with the inputs
    shutil.copyfile(options.pcafile,
                    options.sampleswhitefile)
    datawhitefile = tables.openFile(options.sampleswhitefile, 'r+')
    ncases = datafile.root.inputs.shape[0]
    pcagrp = datawhitefile.root.pca_params
    # get nprincomps!
    varfracs = pcagrp.var_fracs.read()
    if cfg.pca.method == 'blockwise':
        nprincomps = int(cfg.pca.nprinc)
    else:
        # if we do framewise whitening the dimensionality is a multiple of the
        # horizon
        nprincomps = cfg.pca.nprinc * cfg.data.horizon

    filters = tables.Filters(complevel=1, complib='zlib', shuffle=True)
    datawhitefile.createCArray(datawhitefile.root, 'inputs_white',
                               tables.Float32Atom(),
                               shape=(ncases, nprincomps),
                               chunkshape=(100000, nprincomps),
                               filters=filters)
    nbatches = int(np.ceil(ncases / cfg.pca.batchsize))
    for bidx in range(nbatches):
        start = bidx * cfg.pca.batchsize
        end = min((bidx + 1) * cfg.pca.batchsize, ncases)
        print 'whitening batch {0}/{1}...'.format(bidx+1, nbatches)
        if cfg.pca.method == 'blockwise':
            # whitening of the whole block
            datawhitefile.root.inputs_white[start:end] = pca.whiten(
                datafile.root.inputs.read(start, end),
                pcagrp.V.read(),
                pcagrp.m0.read(),
                pcagrp.s0.read(),
                varfracs,
                cfg.pca.retain_var,
                nprincomps=cfg.pca.nprinc,
                batchsize=10000,
                use_gpu=cfg.pca.use_gpu,
                verbose=cfg.pca.verbose)
        else:
            # whitening of the individual frames by reshape before and after
            datawhitefile.root.inputs_white[start:end] = pca.whiten(
                datafile.root.inputs.read(
                    start, end).reshape(
                    ((end - start) * cfg.data.horizon, -1)),
                pcagrp.V.read(),
                pcagrp.m0.read(),
                pcagrp.s0.read(),
                varfracs,
                cfg.pca.retain_var,
                nprincomps=cfg.pca.nprinc,
                batchsize=10000,
                use_gpu=cfg.pca.use_gpu,
                verbose=cfg.pca.verbose).reshape(
                    (end - start, -1))

        datawhitefile.flush()
    datawhitefile.close()
    datafile.close()

# vim: set ts=4 sw=4 sts=4 expandtab:
