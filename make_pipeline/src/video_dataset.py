#!/usr/bin/env python
#-*- coding: utf-8 -*-

import os

import numpy as np
import tables

from videotools import sample_clips_random_from_multiple_videos


class VideoDataset(object):

    def __init__(self, clipsetdir, videodir, labelsfile,
                 fileext='.avi'):
        """Constructor that sets some paths, labels and dataset description
        Args:
            clipsetdir: path to the clipset files
            videodir: path to the video files
            labels: a sequence of labels
            name: a name for the dataset
            fileext: file extension of the video files
        """
        self.clipsetdir = clipsetdir
        self.videodir = videodir
        h5file = tables.openFile(labelsfile, 'r')
        self.labels = h5file.root.labels.read()
        self.ntrain = int(h5file.root.ntrain.read())
        self.ntest = int(h5file.root.ntest.read())
        try:
            self.name = h5file.root.dataset.read()
        except tables.NoSuchNodeError:
            print 'setting dataset name to empty string'
            self.name = ''
        h5file.close()
        self.fileext = fileext
        (self.trainfiles,
         self.trainlabels) = self.get_partition_clips_and_labels('train')
        (self.testfiles,
         self.testlabels) = self.get_partition_clips_and_labels('test')

    def cliplist_from_clipset(self, partition, label=None):
        """Get videolist in given partition, if label is set only for this label
        Args:
            partition: partition ('train' or 'test')
            label: label (if =None, return all in given partition)
        Returns:
            List with video filenames.
        """
        if label is None:
            label = 'actions'
        clipset_file = open(
            os.path.join(
                self.clipsetdir,
                '{0}_{1}.txt'.format(label, partition)),
            'r')
        clip_list = clipset_file.read().splitlines()
        clipset_file.close()
        clip_paths = []
        for line in clip_list:
            line = line.split(' ')
            if len(line) == 1 or line[-1] == '1':
                clip_paths.append('{0}/{1}{2}'.format(
                    self.videodir, line[0], self.fileext))
        return clip_paths

    def get_partition_clips_and_labels(self, partition):
        cliplist = self.cliplist_from_clipset(partition)
        cliplist_per_label = {}
        for label in self.labels:
            cliplist_per_label[label] = self.cliplist_from_clipset(
                partition, label)

        cliplabels = np.zeros((len(cliplist), len(self.labels)),
                                   dtype=np.int64)
        for idx, clip in enumerate(cliplist):
            for i in range(len(self.labels)):
                if clip in cliplist_per_label[self.labels[i]]:
                    cliplabels[idx, i] = 1
        return (cliplist, cliplabels)

    def _getlabelsfromclipset(self, label, partition):
        """Reads labels from file
        Args:
            label: action name or 'actions'
            partition: 'test' or 'train'
        Returns:
            onehot labels
        """
        file_obj = file('{0}/{1}_{2}.txt'.format(
            self.clipsetdir, label, partition), 'r')
        lines = file_obj.read().splitlines()
        labels = np.array([len(line.split(' ')) == 1 or int(line.split(' ')[-1]) > 0
                           for line in file_obj.read().splitlines()],
                          dtype=np.int64)
        file_obj.close()
        return labels

    def gettestlabels(self):
        return self.testlabels

    def gettrainlabels(self):
        return self.trainlabels

    def gettrainfiles(self):
        return self.trainfiles

    def gettestfiles(self):
        return self.testfiles

    def gettrainsamples_random(self, nsamples, framesize, horizon,
                               temporal_subsampling=False):
        return sample_clips_random_from_multiple_videos(
            videolist=self.trainfiles, nsamples=nsamples,
            framesize=framesize, horizon=horizon,
            temporal_subsampling=temporal_subsampling)


# vim: set ts=4 sw=4 sts=4 expandtab:
