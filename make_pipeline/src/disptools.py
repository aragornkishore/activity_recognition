#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy
from PIL import Image
import scipy
import pylab

def scale_to_unit_interval(ndar,eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max()+eps)
    return ndar


def tile_raster_images(X, img_shape, tile_shape,tile_spacing = (0,0), 
              scale_rows_to_unit_interval = True, output_pixel_vals = True):
    """
    Transform an array with one flattened image per row, into an array in 
    which images are reshaped and layed out like tiles on a floor.

    This function is useful for visualizing datasets whose rows are images, 
    and also columns of matrices for transforming those rows 
    (such as the first layer of a neural net).

    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can 
    be 2-D ndarrays or None;
    :param X: a 2-D array in which every row is a flattened image.

    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image

    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)
    
    :param output_pixel_vals: if output should be pixel values (i.e. int8
    values) or floats

    :param scale_rows_to_unit_interval: if the values need to be scaled before
    being plotted to [0,1] or not


    :returns: array suitable for viewing as an image.  
    (See:`PIL.Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.

    """
 
    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as 
    # follows : 
    #
    # out_shape    = [0,0]
    # out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [(ishp + tsp) * tshp - tsp for ishp, tshp, tsp 
                        in zip(img_shape, tile_shape, tile_spacing)]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output numpy ndarray to store the image 
        if output_pixel_vals:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4), dtype='uint8')
        else:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4), dtype=X.dtype)

        #colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0,0,0,255]
        else:
            channel_defaults = [0.,0.,0.,1.]

        for i in xrange(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct 
                # dtype
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                out_array[:,:,i] = numpy.zeros(out_shape,
                        dtype=dt)+channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it 
                # in the output
                out_array[:,:,i] = tile_raster_images(X[i], img_shape, tile_shape, tile_spacing, scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel 
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        dt = X.dtype
        if output_pixel_vals:
            dt = 'uint8'
        out_array = numpy.zeros(out_shape, dtype=dt)


        for tile_row in xrange(tile_shape[0]):
            for tile_col in xrange(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1 
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale_to_unit_interval(X[tile_row * tile_shape[1] + tile_col].reshape(img_shape))
                    else:
                        this_img = X[tile_row * tile_shape[1] + tile_col].reshape(img_shape)
                    # add the slice to the corresponding position in the 
                    # output array
                    c = 1
                    if output_pixel_vals:
                        c = 255
                    out_array[
                        tile_row * (H+Hs):tile_row*(H+Hs)+H,
                        tile_col * (W+Ws):tile_col*(W+Ws)+W
                        ] \
                        = this_img * c
        return out_array

def CreateMovie(filename, plotter, numberOfFrames, fps):
    import os, sys
    import matplotlib.pyplot as plt
    for i in range(numberOfFrames):
        plotter(i)
        fname = '_tmmp%05d.png'%i
        plt.savefig(fname)
        plt.clf()
    #os.system("rm"+" filename"+".mp4")
    #os.system("ffmpeg -b 1800 -i _tmp%05d.png -r " + str(fps) + " "+filename+".mp4")
    os.system("convert -delay 10 -loop 0 _tmmp*.png "+filename+".gif")
    os.system("rm _tmmp*.png")


def dispimsmovie(filename, inv, filters, patchsize, horizon, fps=5,  *args, **kwargs):
    from pylab import imshow,cm
    W = filters
    W = numpy.dot(inv,W).T
    numframes = horizon
    n = W.shape[1]/numframes
    def plotter(i):
        W_ = W[:,i*n:n*(i+1)]
        image = tile_raster_images(W_, img_shape=(patchsize,patchsize), tile_shape=(12,12),tile_spacing = (1,1), scale_rows_to_unit_interval = True, output_pixel_vals = True)
        imshow(image,cmap=cm.gray,interpolation='nearest')
        pylab.axis('off')

    CreateMovie(filename, plotter, numframes, fps)


