#!/usr/bin/env python
#-*- coding: utf-8 -*-

import optparse
import os

import tables

import dictstruct
import litekmeans
import modelloader


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
        '-t', '--kmeanstrainfile', action='store', type='string',
        dest='kmeanstrainfile',
        help='''h5 file containing the kmeans training data''',
        metavar='FILE')

    p.add_option(
        '-k', '--kmeansfile', action='store', type='string',
        dest='kmeansfile',
        help='''h5 file in which the kmeans centroids are stored (CONTENTS WILL
        BE OVERWRITTEN) (required)''',
        metavar='FILE')

    (opts, _) = p.parse_args()
    # check if the required options are specified
    for opt in (opts.cfg, opts.kmeanstrainfile,
                opts.kmeansfile):
        if opt is None:
            p.error('required parameter not provided')

    # check if the specified files really exist
    if not os.path.isfile(opts.cfg):
        p.error('config file {0} does not exist'.format(opts.cfg))
    if not os.path.isfile(opts.kmeanstrainfile):
        p.error('kmeans train data file {0} does not exist'.format(
            opts.kmeanstrainfile))
    return opts

if __name__ == '__main__':
    options  = parse_args()
    cfg = dictstruct.DictStruct(options.cfg)

    mappings_file = tables.openFile(options.kmeanstrainfile)
    mappings = mappings_file.root.kmeanstraindata_white.read()
    mappings_file.close()

    centroids_file = tables.openFile(options.kmeansfile, 'w')
    filters = tables.Filters(complevel=1, complib='zlib', shuffle=True)
    centroids = centroids_file.createCArray(centroids_file.root, 'centroids',
                                tables.Float32Atom(),
                                shape=(cfg.kmeans.numinitializations,
                                       cfg.kmeans.numcentroids,
                                       mappings.shape[1]),
                                filters=filters)
    obj = centroids_file.createCArray(centroids_file.root, 'obj',
                                tables.Float32Atom(),
                                shape=(cfg.kmeans.numinitializations, ))


    for initialization in range(cfg.kmeans.numinitializations):
        print 'running kmeans initialization {0}/{1}'.format(
            initialization+1, cfg.kmeans.numinitializations)
        _, centroids[initialization], obj[initialization] = \
                litekmeans.litekmeans(
                    mappings,
                    cfg.kmeans.numcentroids, max_iter=50)
        centroids_file.flush()
    centroids_file.close()


# vim: set ts=4 sw=4 sts=4 expandtab:
