#!/usr/bin/env python
#-*- coding: utf-8 -*-

from math import ceil
import optparse
import os
import tables

import dictstruct
import video_dataset


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
        help='hdf 5 file to store the samples in (CONTENT WILL BE OVERWRITTEN) (required)',
        metavar='FILE')

    p.add_option(
        '-l', '--labelsfile', action='store', type='string', dest='labelsfile',
        help='hdf5 file containing the labels',
        metavar='FILE')

    # add directory options
    p.add_option(
        '-v', '--videodir', action='store', type='string', dest='videodir',
        help='directory where the videos are stored',
        metavar='DIR')

    p.add_option(
        '-s', '--clipsetdir', action='store', type='string', dest='clipsetdir',
        help='clipset directory',
        metavar='DIR')

    (opts, _) = p.parse_args()
    # check if all required options are specified
    for opt in (opts.cfg, opts.samplesfile, opts.videodir, opts.clipsetdir,
                opts.labelsfile):
        if opt is None:
            p.error('required parameter not provided')

    # check if the files/dirs really exist
    if not os.path.isfile(opts.cfg):
        p.error('config file {0} does not exist'.format(opts.cfg))
    if not os.path.isfile(opts.labelsfile):
        p.error('labels file {0} does not exist'.format(opts.labelsfile))
    if not os.path.isdir(opts.videodir):
        p.error('invalid video dir specified')
    if not os.path.isdir(opts.clipsetdir):
        p.error('invalid clipset dir specified')
    return opts

if __name__ == '__main__':
    options  = parse_args()
    cfg = dictstruct.DictStruct(options.cfg)

    batchsize = min(500000, cfg.trainer.numcases)

    datafile = tables.openFile(options.samplesfile, 'w')
    datafile.createCArray(
        datafile.root, 'inputs', tables.Float32Atom(),
        shape=(cfg.trainer.numcases, cfg.data.horizon * cfg.data.framesize**2))
    dataset = video_dataset.VideoDataset(clipsetdir=options.clipsetdir,
                                         videodir=options.videodir,
                                         labelsfile=options.labelsfile)
    for bidx in range(int(ceil(float(cfg.trainer.numcases / batchsize)))):
        start = bidx*batchsize
        end = min((bidx+1)*batchsize, cfg.trainer.numcases)
        datafile.root.inputs[start:end] = \
            dataset.gettrainsamples_random(
                nsamples=end-start,
                framesize=cfg.data.framesize,
                horizon=cfg.data.horizon,
                temporal_subsampling=cfg.data.temp_subsample)
        datafile.flush()
    datafile.close()

# vim: set ts=4 sw=4 sts=4 expandtab:
