#!/usr/bin/env python
#-*- coding: utf-8 -*-

import os
from optparse import OptionParser


def save_cfg(cfg, cfgfile):
    cfg.dump(cfgfile)

def setup_basic_optparser():
    """Sets up the option parser that is used by all pipeline modules
    Returns:
        An instance of optparse.OptionParser, that parses path options.
    """
    p = OptionParser()
    p.add_option(
        '-c', '--cfg', action='store', type='string', dest='cfg',
        help='path to config file',
        metavar='FILE')
    p.add_option(
        '-d', '--data', action='store', type='string', dest='data',
        help='data directory, usually the parent directory of videos and clipsets directories',
        metavar='DIR')
    p.add_option(
        '-r', '--results', action='store', type='string', dest='results',
        help='results directory',
        metavar='DIR')
    p.add_option(
        '-s', '--clipsets', action='store', type='string', dest='clipsets',
        help='clipsets directory',
        metavar='DIR')
    p.add_option(
        '-v', '--videos', action='store', type='string', dest='videos',
        help='videos directory',
        metavar='DIR')
    p.add_option(
        '-t', '--tmp', action='store', type='string', dest='tmp',
        default='/tmp',
        help='directory, where temporary files should be stored',
        metavar='DIR')
    return p


def parse_args(parser):
    (options, _) = parser.parse_args()
    if options.cfg is None:
        parser.error('no config file specified')
    if not os.path.isfile(options.cfg):
        parser.error('invalid config file specified')
    # check for unspecified or invalid paths
    if options.data is None:
        parser.error('no data path')
    if not os.path.isdir(options.data):
        parser.error('invalid data path specified')

    if options.results is None:
        parser.error('no results path')
    if not os.path.isdir(options.results):
        parser.error('invalid results path specified')

    if options.clipsets is None:
        parser.error('no clipsets path')
    if not os.path.isdir(options.clipsets):
        parser.error('invalid clipsets path specified')

    if options.videos is None:
        parser.error('no videos path')
    if not os.path.isdir(options.videos):
        parser.error('invalid videos path specified')

    if options.tmp is None:
        parser.error('no tmp path (/tmp not recommended)')
    if not os.path.isdir(options.tmp):
        parser.error('invalid tmp path specified')

    return options

# vim: set ts=4 sw=4 sts=4 expandtab:
