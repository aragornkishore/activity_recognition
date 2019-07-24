#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np

import videotools


def cliplist_from_clipset(clipset_filename, avi_dir):
    clipset_file = open(clipset_filename, 'r')
    clip_list = clipset_file.read().splitlines()
    clipset_file.close()
    clip_paths = []
    for line in clip_list:
        line = line.split(' ')
        if line[-1] == '1':
            clip_paths.append('%s/%s.avi' % (avi_dir, line[0]))
    return clip_paths


class Hollywood2Dataset(object):

    def __init__(self, path, clipsetdir='ClipSets', avidir='AVIClips05'):
        self.path = path
        self.clipsetpath = '{0}/{1}'.format(path, clipsetdir)
        self.avipath = '{0}/{1}'.format(path, avidir)
        self.labels = ['AnswerPhone', 'DriveCar', 'Eat', 'FightPerson',
                       'GetOutCar', 'HandShake', 'HugPerson', 'Kiss',
                       'Run', 'SitDown', 'SitUp', 'StandUp']
        self.ntest = 884
        self.ntrain = 823

    def _getlabelsfromclipset(self, label, partition):
        """Reads labels from file
        Args:
            label: action name or 'actions'
            partition: 'test' or 'train'
        Returns:
            onehot labels
        """
        file_obj = file('{0}/{1}_{2}.txt'.format(self.clipsetpath, label, partition), 'r')
        labels = np.array([int(line.split(' ')[-1]) > 0
                           for line in file_obj.read().splitlines()],
                          dtype=np.int64)
        file_obj.close()
        return labels

    def gettestlabels(self):
        testlabels = np.zeros((self.ntest, len(self.labels)), dtype=np.int64)
        for i in range(len(self.labels)):
            testlabels[:, i] = self._getlabelsfromclipset(self.labels[i], 'test')
        return testlabels

    def gettrainlabels(self):
        trainlabels = np.zeros((self.ntrain, len(self.labels)), dtype=np.int64)
        for i in range(len(self.labels)):
            trainlabels[:, i] = self._getlabelsfromclipset(self.labels[i], 'train')
        return trainlabels

    def gettrainsamples_random(self, nsamples, framesize, horizon, temporal_subsampling=False):
        videolist = cliplist_from_clipset(
            '{0}/{1}'.format(self.clipsetpath, 'actions_train.txt'),
            self.avipath)
        return videotools.sample_clips_random_from_multiple_videos(
            videolist=videolist, nsamples=nsamples,
            framesize=framesize, horizon=horizon,
            temporal_subsampling=temporal_subsampling)

    def gettestsamples_dense(self):
        pass

if __name__ == '__main__':
    h2 = Hollywood2Dataset('/home/vincent/data/hollywood2_isa')
    testlabels = h2.gettestlabels()
    trainlabels = h2.gettrainlabels()

# vim: set ts=4 sw=4 sts=4 expandtab:
