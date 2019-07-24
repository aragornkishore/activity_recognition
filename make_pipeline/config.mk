# config.mk

# Directories

# directory where results are stored
RESULTSDIR=$(CURDIR)

# the source directory
SRCDIR=$(RESULTSDIR)/src

# clipset directory
CLIPSETDIR=$(HOME)/software/python_env/bin/.exclude/data/kth_sub/ClipSets

# labels file
LABELSFILE=kth_labels.h5

# video directory
VIDEODIR=$(HOME)/software/python_env/bin/.exclude/data/kth_sub

# tmp directory to be used
TMP_DIR=tmp


# Path to the config file
CFGFILE=$(CURDIR)/pipeline.cfg

# short hand fore cfg param
CFGPARAM=--cfg=$(CFGFILE)


# Target files

# file in which the samples are stored
SAMPLES_FILE=$(TMP_DIR)/samples.h5

# samples white file includes the pca params, so we save them in tmp
PCA_FILE=$(RESULTSDIR)/pca_params.h5

# file in which the whitened samples are stored
SAMPLES_WHITE_FILE=$(TMP_DIR)/samples_white.h5

# model param file
MODELFILE=$(RESULTSDIR)/model_params.npy

# trainer param file
TRAINERFILE=$(RESULTSDIR)/trainer_params.h5

# kmeans train inputs file
KMEANSTRAINFILE=$(TMP_DIR)/kmeans_traindata.h5

# kmeans centroids file
KMEANSFILE=$(RESULTSDIR)/kmeans_centroids.h5

# classifier train and test data files
CLASSIFIER_TRAINFILE=$(RESULTSDIR)/classifier_traindata.h5
CLASSIFIER_TESTFILE=$(RESULTSDIR)/classifier_testdata.h5

REPORTFILE=$(RESULTSDIR)/results.txt

NTHREADS=8

# Shell to be used
SHELL = /bin/bash

# Python command to be used
PYTHON=python -u

