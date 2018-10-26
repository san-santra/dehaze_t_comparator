'''
Copyright (C) 2018  Sanchayan Santra

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''

from skimage import img_as_float, filters
import skimage.io as skio
from skimage.color import rgb2gray
import cv2
import time
import numpy as np
import os


def discard_patch(patch, var_thr, edge_thr):
    '''
    Discard the patch if
    1. it's smooth
    2. contains edge ( gradient magnitude is high )
    '''
    # these two checks are done on the grayscale version of the patch
    gray_patch = rgb2gray(patch)

    var = np.std(gray_patch)
    if var < var_thr:
        # implies smooth patch
        return True

    edge = filters.sobel(gray_patch)
    if np.any(edge > edge_thr):
        # checking with np.any will cause some non-edge patches to be discarded
        return True

    return False


if __name__ == '__main__':
    # parameters
    patch_X = 10
    patch_Y = 10
    stride_X = patch_X/2
    stride_Y = patch_Y/2

    var_thr = 0.02
    edge_thr = 0.5

    clean_dir = './data/training_images'
    clean_images = os.listdir(clean_dir)

    outf_patches = 'data/clustered_patches.npy'

    # first count the patches, then allocate memory
    # although takes more time but space efficient (stops MemoryError)
    numpatch = 0  # for counting
    print "Counting patches"
    for i in xrange(len(clean_images)):
        # need to open the image because some of the patches will be discarded
        # and it's better to give the network values between 0 and 1
        c_im = img_as_float(skio.imread(os.path.join('.', clean_dir,
                                                     clean_images[i])))

        print "(%d/%d)-%s" % (i + 1, len(clean_images), clean_images[i])

        [nrow, ncol, _] = c_im.shape
        x = np.r_[:nrow - patch_X:stride_X]
        y = np.r_[:ncol - patch_Y:stride_Y]

        # now get the patch
        for idx in xrange(x.size):
            for idy in xrange(y.size):
                # it's better to copy as indexing returns reference
                patch = c_im[x[idx]:x[idx] + patch_X,
                             y[idy]:y[idy]+patch_Y, :].copy()

                # should this be discarded ?
                if not discard_patch(patch, var_thr, edge_thr):
                    # if not increase the patch count
                    numpatch += 1

    print "counting done"
    print "allocating memory"
    # Counting done, allocate memory
    # need to store the RGB values and the gradient
    feature_vec = np.zeros((numpatch, (patch_X*patch_Y*3) +
                            (patch_X-2)*(patch_Y-2)*2), dtype='float32')

    # store the patches for the clustering, with provision to convert
    # them back to the format for the CNN
    
    # Now get the patches
    c_p_idx = 0
    for i in xrange(len(clean_images)):
        c_im = img_as_float(skio.imread(os.path.join('.', clean_dir,
                                                     clean_images[i])))

        print "(%d/%d)-%s" % (i + 1, len(clean_images), clean_images[i])

        [nrow, ncol, _] = c_im.shape
        x = np.r_[:nrow - patch_X:stride_X]
        y = np.r_[:ncol - patch_Y:stride_Y]

        # now get the patch
        for idx in xrange(x.size):
            for idy in xrange(y.size):
                # it's better to copy as indexing returns reference
                patch = c_im[x[idx]:x[idx] + patch_X,
                             y[idy]:y[idy]+patch_Y, :].copy()

                # should this patch be discarded ?
                if discard_patch(patch, var_thr, edge_thr):
                    continue

                # sobel requires gray image
                gray_patch = rgb2gray(patch)

                # need to ignore the borders, they are 0
                g_x = filters.sobel_h(gray_patch)[1:-1, 1:-1]
                g_y = filters.sobel_v(gray_patch)[1:-1, 1:-1]

                feature_vec[c_p_idx, :] \
                    = np.concatenate((patch.flatten(), g_x.flatten(),
                                      g_y.flatten()))
                c_p_idx += 1

    print feature_vec.shape

    # from scipy.cluster.vq import whiten, kmeans
    # scipy giving memory error with full data

    # just check the output once with small number of images
    n_cluster = 100000
    n_attempt = 5
    
    print "Starting the clustering"
    start_time = time.time()
    criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 10, 1e-05)
    _, label, center = cv2.kmeans(feature_vec, n_cluster, criteria, n_attempt,
                                  cv2.KMEANS_PP_CENTERS)
    end_time = time.time()
    print "Exec time: {}".format(end_time - start_time)

    np.save(outf_patches, center)
    
