#!/usr/bin/env python
#-*- coding: utf-8 -*-

from __future__ import division

import optparse
import os
import Queue
import threading
import time

import numpy as np
import pca
import tables

import debugtools
import dictstruct
import modelloader
import video_dataset
from videotools import (get_num_subblocks,
                        sample_clips_dense_from_multiple_videos,
                        sample_clips_random)

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
        help='''h5 file in which the model params are stored (required)''',
        metavar='FILE')

    p.add_option(
        '-p', '--pcafile', action='store', type='string',
        dest='pcafile',
        help='''h5 file in which pca parameters are stored''',
        metavar='FILE')

    p.add_option(
        '-t', '--kmeanstrainfile', action='store', type='string',
        dest='kmeanstrainfile',
        help='''h5 file in which the kmeans train inputs are stored (CONTENTS
        WILL BE OVERWRITTEN) (required)''',
        metavar='FILE')
    p.add_option(
        '-n', '--nthreads', action='store', type='int',
        dest='nthreads',
        help='''number of threads to be used''',
        metavar='NUMBER')

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
    # check if the required options are specified
    for opt in (opts.cfg, opts.sampleswhitefile,
                opts.modelfile, opts.kmeanstrainfile,
                opts.nthreads, opts.pcafile,
                opts.labelsfile, opts.videodir,
                opts.clipsetdir):
        if opt is None:
            p.error('required parameter not provided')

    # check if the specified files really exist
    if not os.path.isfile(opts.cfg):
        p.error('config file {0} does not exist'.format(opts.cfg))
    if not os.path.isfile(opts.sampleswhitefile):
        p.error('whitened samples file {0} does not exist'.format(
            opts.sampleswhitefile))
    if not os.path.isfile(opts.labelsfile):
        p.error('labels file {0} does not exist'.format(opts.labelsfile))
    if not os.path.isfile(opts.modelfile):
        p.error('model file {0} does not exist'.format(
            opts.modelfile))
    if not os.path.isfile(opts.pcafile):
        p.error('pca file {0} does not exist'.format(
            opts.pcafile))
    if not os.path.isdir(opts.videodir):
        p.error('video dir {0} does not exist'.format(
            opts.videodir))
    if not os.path.isdir(opts.clipsetdir):
        p.error('clipset dir {0} does not exist'.format(
            opts.clipsetdir))
    return opts

if __name__ == '__main__':
    options  = parse_args()
    cfg = dictstruct.DictStruct(options.cfg)

    datawhitefile = tables.openFile(options.sampleswhitefile, 'r')
    pca_params = datawhitefile.root.pca_params

    V = pca_params.V.read()
    m0 = pca_params.m0.read()
    s0 = pca_params.s0.read()
    retain_var = cfg.pca.retain_var
    nprinc = cfg.pca.nprinc
    var_fracs = pca_params.var_fracs.read()
    del pca_params
    numvis = datawhitefile.root.inputs_white.shape[1]
    datawhitefile.close()

    model = modelloader.load_model(cfg, options.modelfile, numvis)
    model_mutex = threading.Lock()

    dataset = video_dataset.VideoDataset(clipsetdir=options.clipsetdir,
                                         videodir=options.videodir,
                                         labelsfile=options.labelsfile)

    #FIXME: save pca_params of superblocks to pca_params file
    kmeans_numcases = cfg.kmeans.numcases
    filelist = dataset.gettrainfiles()


    nsamples_per_file = np.ceil(
        kmeans_numcases / len(filelist))
    print 'will sample {0} superblocks from each of {1} train videos'.format(
        int(nsamples_per_file), len(filelist))

    kmeanstrainfile = tables.openFile(options.kmeanstrainfile, 'w')
    kmeanstraindata = kmeanstrainfile.createCArray(
        kmeanstrainfile.root,
        'kmeanstraindata',
        tables.Float32Atom(),
        shape=(nsamples_per_file * len(filelist),
            get_num_subblocks(
                superblock_framesize = cfg.data.superblock_framesize,
                superblock_horizon = cfg.data.superblock_horizon,
                framesize = cfg.data.framesize,
                horizon = cfg.data.horizon,
                stride = cfg.data.stride
            ) * cfg.model.nummap))
    #kmeanstraindata = np.zeros(
    #    (nsamples_per_file * len(filelist),
    #        get_num_subblocks(
    #            superblock_framesize = cfg.data.superblock_framesize,
    #            superblock_horizon = cfg.data.superblock_horizon,
    #            framesize = cfg.data.framesize,
    #            horizon = cfg.data.horizon,
    #            stride = cfg.data.stride
    #        ) * cfg.model.nummap), dtype=np.float32)

    print 'shape of kmeanstraindata: %s' % (kmeanstraindata.shape, )

    input_mutex = threading.Lock()

    # create a jobqueue, job: tuple of index and path to file from which to
    # sample
    jobs = Queue.Queue()
    for i in range(len(filelist)):
        jobs.put((i, filelist[i]))
    print 'number of jobs: {0}'.format(jobs.qsize())

    assert jobs.empty() == False

    # each thread gets its own space to put tmp data
    features = []
    for i in range(options.nthreads):
        features.append(None)

    start = time.time()

    def _sample_subblocks(threadidx):
        """Samples superblocks, concatenates mappings of the contained blocks
        Returns:
            The cluster assignments
        """
        while True:
            try:
                job = jobs.get_nowait()
            except Queue.Empty:
                print "QUEUE IS EMPTY"
                return
            print '{0}: processing job {1}'.format(threadidx, job[0])

            # sample random superblocks
            features[threadidx] = sample_clips_random(
                video=job[1],
                framesize=cfg.data.superblock_framesize,
                horizon=cfg.data.superblock_horizon,
                temporal_subsampling=cfg.data.temp_subsample,
                nsamples=nsamples_per_file)
            if np.any(np.isnan(features[threadidx])):
                raise ValueError('nan detected in random samples')

            # get blocks from superblocks
            features[threadidx] = sample_clips_dense_from_multiple_videos(
                features[threadidx].reshape((features[threadidx].shape[0],
                                            cfg.data.superblock_horizon,
                                            cfg.data.superblock_framesize,
                                            cfg.data.superblock_framesize)),
                framesize=cfg.data.framesize,
                horizon=cfg.data.horizon,
                temporal_subsampling=cfg.data.temp_subsample,
                stride=cfg.data.stride, verbose=False)
            if np.any(np.isnan(features[threadidx])):
                raise ValueError('nan detected in sub-samples')

            # whiten the samples
            if cfg.pca.method == 'blockwise':
                features[threadidx] = pca.whiten(
                    data=features[threadidx], V=V, m0=m0, s0=s0,
                    var_fracs=var_fracs, nprincomps=nprinc,use_gpu=False,
                    retain_var=cfg.pca.retain_var)
            else:
                features[threadidx] = pca.whiten(
                    data=features[threadidx].reshape(
                        (features[threadidx].shape[0] * cfg.data.horizon,
                            -1)),
                    V=V, m0=m0, s0=s0,
                    var_fracs=var_fracs, nprincomps=nprinc, use_gpu=False,
                    retain_var=cfg.pca.retain_var).reshape(
                        (features[threadidx].shape[0], -1))

            if np.any(np.isnan(features[threadidx])):
                raise ValueError('nan detected in whitened samples')

            # get the mappings
            model_mutex.acquire()
            features[threadidx] = model.mappingsNonoise_batchwise(
                features[threadidx], batchsize=1000)
            model_mutex.release()
            if np.any(np.isnan(features[threadidx])):
                raise ValueError('nan detected in mappings')
            # concatenate the mappings
            features[threadidx] = features[threadidx].reshape((
                nsamples_per_file, -1))

            input_mutex.acquire()
            kmeanstraindata[
                job[0] * nsamples_per_file:(job[0] + 1) * nsamples_per_file, :] =\
                    features[threadidx]
            # thread 0 should flush
            if threadidx == 0:
                kmeanstrainfile.flush()
            input_mutex.release()


    # start the threads
    threads = []
    for threadidx in range(options.nthreads):
        threads.append(threading.Thread(target=_sample_subblocks,
                                        args=(threadidx, )))
        threads[-1].start()
    # check if all threads are finished
    while True:
        all_finished = True
        for threadidx in range(len(threads)):
            if threads[threadidx].is_alive():
                all_finished = False
        if all_finished:
            print 'all threads finished'
            del features
            break
        else:
            time.sleep(1)

    # dump samples to file
    #kmeanstrainfile = tables.openFile(options.kmeanstrainfile, 'w')
    #kmeanstrainfile.createArray(
    #    kmeanstrainfile.root,
    #    'kmeanstraindata', kmeanstraindata[:kmeans_numcases])
    print 'selecting samples for pca'

    # compute pca params and also dump them into the file
    print 'computing pca params'
    V_sb, W_sb, m0_sb, s0_sb, var_fracs_sb = pca.pca(
        kmeanstrainfile.root.kmeanstraindata[:cfg.pca.numsamples])
    print 'dumping pca params'

    # FIXME: hardcoded!!!
    
    nprincomps_sb = 100

    retained_var_sb = var_fracs_sb[nprincomps_sb]

    pcagrp = kmeanstrainfile.createGroup(kmeanstrainfile.root,
                                         'pca_params')
    kmeanstrainfile.createArray(pcagrp,
                                'V', V_sb)
    kmeanstrainfile.createArray(pcagrp,
                                'W', W_sb)
    kmeanstrainfile.createArray(pcagrp,
                                'm0', m0_sb)
    kmeanstrainfile.createArray(pcagrp,
                                's0', s0_sb)
    kmeanstrainfile.createArray(pcagrp,
                                'var_fracs', var_fracs_sb)
    kmeanstrainfile.createArray(pcagrp, 'nprincomps', nprincomps_sb)
    kmeanstrainfile.createArray(pcagrp, 'retained_var', var_fracs_sb[nprincomps_sb])
    kmeanstrainfile.flush()


    # whiten the samples
    print 'whitening kmeans samples'

    print 'kmeans_numcases: %r' % (kmeans_numcases, )
    print 'nprincomps_sb: %r' % (nprincomps_sb, )
    kmeanstrainfile.createArray(
        kmeanstrainfile.root,
        'kmeanstraindata_white',
        np.zeros((kmeans_numcases, nprincomps_sb), dtype=np.float32))

    nbatches = int(np.ceil(kmeans_numcases/ cfg.pca.batchsize))
    for bidx in range(nbatches):
        start = bidx * cfg.pca.batchsize
        end = min((bidx + 1) * cfg.pca.batchsize, kmeans_numcases)
        print 'whitening batch {0}/{1}...'.format(bidx+1, nbatches)
        kmeanstrainfile.root.kmeanstraindata_white[start:end] = pca.whiten(
            kmeanstrainfile.root.kmeanstraindata.read(start, end),
            V_sb, m0_sb, s0_sb, var_fracs_sb, retained_var_sb,
            batchsize=5000,
            use_gpu=cfg.pca.use_gpu,
            verbose=cfg.pca.verbose)[:,:nprincomps_sb]

        kmeanstrainfile.flush()
    # delete the unwhitened samples
    kmeanstrainfile.removeNode(kmeanstrainfile.root, 'kmeanstraindata')
    kmeanstrainfile.close()

    print 'time spent (): {0:d}'.format(int(time.time() - start))

# vim: set ts=4 sw=4 sts=4 expandtab:
