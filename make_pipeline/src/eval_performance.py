#!/usr/bin/env python
#-*- coding: utf-8 -*-

import inspect
import os
import sys

import numpy as np
import scipy.io

from chi2 import chi2
sys.path.insert(0,
    os.path.join(
        os.path.realpath(os.path.abspath('.')),
        'libsvm-3.13/python'))

sys.path.insert(0,
    os.path.join(
        os.path.realpath(os.path.abspath('.')),
        'libsvm-3.13/python'))
import svmutil

def compute_kernel_matrices(train_inputs, test_inputs):
    """Computes the kernel matrices for train and test inputs
    Args:
        train_inputs: training inputs (numpy array, stacked row-wise)
        test_inputs: test inputs (numpy array, stacked row-wise)
    Returns:
        tuple of kernel matrices (one for train and one for test inputs)
    """
    kernel_train = chi2.chi2_kernel(train_inputs,train_inputs)
    kernel_train = np.concatenate((np.arange(1,kernel_train[0].shape[0]+1).reshape(-1,1), kernel_train[0]),1)

    kernel_test = chi2.chi2_kernel(test_inputs,train_inputs)
    kernel_test = np.concatenate((np.arange(1,kernel_test[0].shape[0]+1).reshape(-1,1), kernel_test[0]),1)

    print 'shape of kernel_train[0]: %s' % (kernel_train.shape, )
    print 'type of kernel_train[0]: %s' % (type(kernel_train), )
    print 'shape of kernel_test[0]: %s' % (kernel_test.shape, )
    print 'type of kernel_test[0]: %s' % (type(kernel_test), )
    

    return (kernel_train, kernel_test)

def compute_aps(confidences, gtlabels):
    #each line in confidences and gtlabels corresponds to one input
    #each column corresponds to one class
    aps = 0.0
    # compute precision/recall
    si = np.argsort(-confidences, 0)
    
    tp = (gtlabels[si] == 1).astype("int")
    fp = (gtlabels[si] == 0).astype("int")
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / (gtlabels).sum().astype("float")
    prec = tp / (fp+tp).astype("float")
    # compute average precision
    for t in np.linspace(0.0, 1.0, 11):
        try:
            p = prec[rec >= t].max()
        except:
            continue
        aps = aps + p/11.
    return aps

def classify(train_inputs, train_outputs, test_inputs, test_outputs):
    kernel_train, kernel_test = \
        compute_kernel_matrices(train_inputs, test_inputs)
       
    counter = 0
    average_ap = 0
    average_acc = 0
    for label in range(train_outputs.shape[1]):
        n_tot = train_inputs.shape[0]
        n_pos = train_outputs[:, label].sum()
        n_neg = n_tot - n_pos
        w_pos = np.float32(n_tot)/(2*n_pos)
        w_neg = np.float32(n_tot)/(2*n_neg)
        option_string = '-t 4 -q -s 0 -b 1 -c %f -w1 %f -w0 %f' % (100, w_pos, w_neg)

        model = svmutil.svm_train(
            train_outputs[:, label].tolist(),
            kernel_train.tolist(),
            option_string)

        _, accuracy, prob_estimates = svmutil.svm_predict(
            test_outputs[:, label].tolist(),
            kernel_test.tolist(), model, '-b  1')
        ap = compute_aps(np.array(prob_estimates)[:,np.where(np.asarray(model.get_labels())==1)], 
            test_outputs[:,label])
        average_ap += ap
        average_acc += accuracy[0]
        counter += 1

        print 'label = %d, ap = %f, w_neg = %f, w_pos = %f\n' % (
            label, ap, w_neg, w_pos)

    mean_ap = np.float32(average_ap) / train_outputs.shape[1]
    mean_acc = np.float32(average_acc) / train_outputs.shape[1]
    print 'mean_ap = %f, mean_acc = %f\n' % (mean_ap, mean_acc)

if __name__ == '__main__':
    train_data = np.load('/home/kishore/research/results0083/more_samples/MM/overlap/classifier_traindata.npz')
    test_data = np.load('/home/kishore/research/results0083/more_samples/MM/overlap/classifier_testdata.npz')
    print 'type of train_data["inputs"]: %s' % (type(train_data["inputs"]), )
    print 'shape of train_data["inputs"]: %s' % (train_data["inputs"].shape, )
    print 'type of train_data["inputs"].tolist(): %s' % (type(train_data["inputs"].astype(np.float64).tolist()[0]), )
    classify(train_data['inputs'], train_data['outputs'],
             test_data['inputs'], test_data['outputs'])

# vim: set ts=4 sw=4 sts=4 expandtab: