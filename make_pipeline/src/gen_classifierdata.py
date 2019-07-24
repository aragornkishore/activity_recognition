#!/usr/bin/env python
#-*- coding: utf-8 -*-

import optparse
import os
import Queue
import threading
import time

import numpy as np
import tables

import commontools
import dictstruct
import video_dataset
import modelloader
import pca
from videotools import (get_num_subblocks,
                        sample_clips_dense,
                        sample_clips_dense_from_multiple_videos)


def get_dense_samples(cfg, nthreads, filelist,
                      kmeansfile, sampleswhitefile,
                      modelfile, kmeanstrainfile):
    """generates dense samples and processes them
    """
    start = time.time()
    centroidsfile = tables.openFile(kmeansfile, 'r')
    centroids = centroidsfile.root.centroids.read()
    centroidsfile.close()
    datawhitefile = tables.openFile(sampleswhitefile, 'r')
    V = datawhitefile.root.pca_params.V.read()
    m0 = datawhitefile.root.pca_params.m0.read()
    s0 = datawhitefile.root.pca_params.s0.read()
    var_fracs = datawhitefile.root.pca_params.var_fracs.read()
    numvis = datawhitefile.root.inputs_white.shape[1]
    datawhitefile.close()
    kmeanstrainfile = tables.openFile(kmeanstrainfile, 'r')

    V_sb = kmeanstrainfile.root.pca_params.V.read()
    m0_sb = kmeanstrainfile.root.pca_params.m0.read()
    s0_sb = kmeanstrainfile.root.pca_params.s0.read()
    var_fracs_sb = kmeanstrainfile.root.pca_params.var_fracs.read()
    nprincomps_sb = kmeanstrainfile.root.pca_params.nprincomps.read()
    retained_var_sb = var_fracs_sb[nprincomps_sb]
    kmeanstrainfile.close()

    n_subblocks = get_num_subblocks(
        superblock_framesize = cfg.data.superblock_framesize,
        superblock_horizon = cfg.data.superblock_horizon,
        framesize = cfg.data.framesize,
        horizon = cfg.data.horizon,
        stride = cfg.data.stride)

    # load data
    model = modelloader.load_model(cfg, modelfile, numvis)
    model_mutex = threading.Lock()

    c2 = np.zeros((cfg.kmeans.numinitializations, cfg.kmeans.numcentroids, 1),
                  dtype=np.float32)
    for initialization in range(cfg.kmeans.numinitializations):
        c2[initialization] = 0.5 *\
            np.sum(centroids[initialization] ** 2, axis=1).reshape((-1, 1))

    inputs = np.empty((cfg.kmeans.numinitializations,
                       len(filelist), centroids.shape[1]), dtype=np.float32)
    jobs = Queue.Queue()
    for i in range(len(filelist)):
        jobs.put(i)

    input_mutex = threading.Lock()
    features = []
    features_sb = []
    features_tmp = []
    for i in range(nthreads):
        features.append(None)
        features_sb.append(None)
        features_tmp.append(None)

    def assign_centroid_indices(threadidx):
        """Samples, preprocesses, gets mappings and assigns cluster centers
        Returns:
            The cluster assignments
        """
        while True:
            try:
                job = jobs.get_nowait()
                print '{0}: processing file {1}/{2}'.format(
                    threadidx, job + 1, len(filelist))
            except Queue.Empty:
                return
            features[threadidx] = sample_clips_dense(
                video=filelist[job],
                framesize=cfg.data.superblock_framesize,
                horizon=cfg.data.superblock_horizon,
                temporal_subsampling=cfg.data.temp_subsample,
                stride=cfg.data.superblock_stride)

            features[threadidx] = features[threadidx].reshape(
                (-1, cfg.data.superblock_horizon,
                 cfg.data.superblock_framesize,
                 cfg.data.superblock_framesize))


            features_sb[threadidx] = np.zeros(
                (features[threadidx].shape[0],
                 cfg.model.nummap * n_subblocks),
                dtype=np.float32)
            features_tmp[threadidx] = sample_clips_dense_from_multiple_videos(
                features[threadidx],
                framesize=cfg.data.framesize,
                horizon=cfg.data.horizon,
                temporal_subsampling=cfg.data.temp_subsample,
                stride = cfg.data.stride,
                verbose=False)


            if cfg.pca.method == 'blockwise':
                features_tmp[threadidx] = pca.whiten(
                    data=features_tmp[threadidx], V=V, m0=m0, s0=s0,
                    var_fracs=var_fracs, use_gpu=False,
                    retain_var=cfg.pca.retain_var,
                    nprincomps=cfg.pca.nprinc)
            else:
                features_tmp[threadidx] = pca.whiten(
                    data=features_tmp[threadidx].reshape(
                        (features_tmp[threadidx].shape[0] * cfg.data.horizon,
                         -1)),
                    V=V, m0=m0, s0=s0,
                    var_fracs=var_fracs, use_gpu=False,
                    retain_var=cfg.pca.retain_var,
                    nprincomps=cfg.pca.nprinc).reshape(
                        (features_tmp[threadidx].shape[0], -1))


            model_mutex.acquire()
            features_tmp[threadidx] = model.mappingsNonoise_batchwise(
                features_tmp[threadidx], 100)
            model_mutex.release()
            features_sb[threadidx] = features_tmp[threadidx].reshape(
                (-1, cfg.model.nummap * n_subblocks))
            features[threadidx] = features_sb[threadidx]

            MM = features[threadidx].sum(1).reshape(
                (-1, 1)) / (n_subblocks * cfg.model.nummap)

            features[threadidx] = pca.whiten(
                data=features[threadidx], V=V_sb, m0=m0_sb, s0=s0_sb,
                var_fracs=var_fracs_sb, use_gpu=False,
                retain_var=np.float32(retained_var_sb),
                nprincomps=nprincomps_sb)
            if features[threadidx].shape[1] > nprincomps_sb:
                #print "WARNING: PCA.WHITEN RETURNED MORE PRINCIPAL COMPONENTS THAN SPECIFIED"
                features[threadidx] = features[threadidx][:,:nprincomps_sb]


            for initialization in range(cfg.kmeans.numinitializations):
                #print 'shape of centroids: %s' % (centroids.shape, )
                c3 = 0.5 * np.sum(features[threadidx]**2, axis=1).reshape((1,-1))
                c3 = c2[initialization] + c3

                clustered = commontools.onehot(
                    np.argmin(
                        c3 -
                        np.dot(centroids[initialization], features[threadidx].T), axis=0),
                              centroids.shape[1]) * MM
                clustered = clustered.sum(0).reshape((1, -1))
                input_mutex.acquire()
                inputs[initialization, job, :] = clustered / np.float32(np.sum(clustered))
                input_mutex.release()

    threads = []
    for threadidx in range(nthreads):
        threads.append(threading.Thread(target=assign_centroid_indices, args=(threadidx, )))
        threads[-1].start()
    while True:
        all_finished = True
        for threadidx in range(len(threads)):
            if threads[threadidx].is_alive():
                all_finished = False
        if all_finished:
            print 'all threads finished'
            break
        else:
            time.sleep(1)

    print 'time spent in get_dense_samples(): {0:d}'.format(int(time.time() - start))
    return inputs.astype(np.float32)

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
        '-p', '--kmeanstrainfile', action='store', type='string',
        dest='kmeanstrainfile',
        help='''h5 file in which the pca params of the superblocks are stored (required)''',
        metavar='FILE')

    p.add_option(
        '-k', '--kmeansfile', action='store', type='string',
        dest='kmeansfile',
        help='''h5 file in which the kmeans centroids are stored (CONTENTS WILL
        BE OVERWRITTEN) (required)''',
        metavar='FILE')

    p.add_option(
        '-t', '--classifiertrainfile', action='store', type='string',
        dest='classifiertrainfile',
        help='''h5 file in which the classifier traindata is stored (CONTENTS WILL
        BE OVERWRITTEN) (required)''',
        metavar='FILE')

    p.add_option(
        '-e', '--classifiertestfile', action='store', type='string',
        dest='classifiertestfile',
        help='''h5 file in which the classifier testdata is stored (CONTENTS WILL
        BE OVERWRITTEN) (required)''',
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
    # check if the required options are specified
    for opt in (opts.cfg, opts.sampleswhitefile,
                opts.modelfile, opts.kmeansfile,
                opts.classifiertrainfile, opts.classifiertestfile,
                opts.clipsetdir, opts.videodir, opts.labelsfile,
                opts.kmeanstrainfile):
        if opt is None:
            p.error('required parameter not provided')

    # check if the specified files/dirs really exist
    if not os.path.isfile(opts.cfg):
        p.error('config file {0} does not exist'.format(opts.cfg))
    if not os.path.isfile(opts.labelsfile):
        p.error('labels file {0} does not exist'.format(opts.labelsfile))
    if not os.path.isfile(opts.sampleswhitefile):
        p.error('whitened samples file {0} does not exist'.format(
            opts.sampleswhitefile))
    if not os.path.isfile(opts.modelfile):
        p.error('model file {0} does not exist'.format(
            opts.modelfile))
    if not os.path.isfile(opts.kmeanstrainfile):
        p.error('kmeans traindata file {0} does not exist'.format(
            opts.kmeanstrainfile))
    if not os.path.isfile(opts.kmeansfile):
        p.error('kmeans file {0} does not exist'.format(
            opts.kmeansfile))
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
    dataset = video_dataset.VideoDataset(clipsetdir=options.clipsetdir,
                                         videodir=options.videodir,
                                         labelsfile=options.labelsfile)

    trainfilelist = dataset.gettrainfiles()
    testfilelist = dataset.gettestfiles()

    # create hdf5 files
    traindata_file = tables.openFile(options.classifiertrainfile, 'w')
    testdata_file = tables.openFile(options.classifiertestfile, 'w')
    # create arrays for the inputs and outputs
    traindata_file.createArray(
        traindata_file.root,
        'traininputs',
        get_dense_samples(
            cfg, nthreads=8, filelist=trainfilelist,
            kmeansfile=options.kmeansfile,
            sampleswhitefile=options.sampleswhitefile,
            modelfile=options.modelfile,
            kmeanstrainfile=options.kmeanstrainfile))
    traindata_file.createArray(
        traindata_file.root,
        'trainoutputs',
        dataset.gettrainlabels())
    traindata_file.flush()
    testdata_file.createArray(
        testdata_file.root,
        'testinputs', get_dense_samples(
            cfg, nthreads=8, filelist=testfilelist,
            kmeansfile=options.kmeansfile,
            sampleswhitefile=options.sampleswhitefile,
            modelfile=options.modelfile,
            kmeanstrainfile=options.kmeanstrainfile))
    testdata_file.flush()
    testdata_file.createArray(
        testdata_file.root,
        'testoutputs',
        dataset.gettestlabels())
    traindata_file.flush()
    testdata_file.flush()
    traindata_file.close()
    testdata_file.close()


# vim: set ts=4 sw=4 sts=4 expandtab:
