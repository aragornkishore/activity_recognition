#!/usr/bin/env python
#-*- coding: utf-8 -*-

from __future__ import print_function
import os

labels = ['boxing', 'handclapping', 'handwaving',
          'jogging', 'running', 'walking']

# division into train / test sets is the same as in
# Wang et al., "Evaluation of local spatio-temporal features for action recognition"
person_indices = {
    'train': [1, 4, 11, 12, 13, 14, 15, 16, 17,
              18, 19, 20, 21, 23, 24, 25],
    'test':  [2, 3, 5, 6, 7, 8, 9, 10, 22]
}

try:
    os.mkdir('ClipSets')
except OSError, e:
    print('{0}'.format(e))

kth_sets = {}

for setname in ('train', 'test'):
    kth_sets[setname] = {}
    for label in labels:
        for pidx in person_indices[setname]:
            for scenario_idx in range(4):
                fname = '{0}/person{1:02d}_{0}_d{2}_uncomp'.format(
                    label, pidx, scenario_idx)
                kth_sets[setname][fname] = label


for setname in ('train', 'test'):
    for label in labels:
        fp = open('ClipSets/{0}_{1}.txt'.format(label, setname), 'w')
        for fname in sorted(kth_sets[setname].keys()):
            if kth_sets[setname][fname] == label:
                indicator = 1
            else:
                indicator = -1
            print('{0}  {1}'.format(fname, indicator), file=fp)
        fp.close()

    fpall = open('ClipSets/actions_{0}.txt'.format(setname), 'w')
    for fname in sorted(kth_sets[setname].keys()):
        print('{0}  1'.format(fname), file=fpall)
    fpall.close()





# vim: set ts=4 sw=4 sts=4 expandtab:
