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

# Binary search code
import os
import sys
import time
import numpy as np
from keras.models import load_model
from skimage import img_as_float
import skimage.io as skio
from scipy import sparse
from scipy.sparse import linalg as splinalg

# local lib
from lib import compute_A_DCP, get_laplacian_4neigh, get_patch_indices
from lib import dehaze_patch
from lib import patch_variance_test, patch_edge_test, patch_airlight_angle_test


def dehaze_comp_serial(im, A, comp_model, patch_X, patch_Y, stride_X, stride_Y,
                       var_thr, edge_thr, angle_thr, angle_test_pixel_frac,
                       alpha, start_t, end_t, t_tol):
    '''
    Will return the predicted t(x) and the dehazed image using binary search
    '''

    [nrow, ncol, nch] = im.shape
    numpixel = nrow * ncol
    
    # preallocate
    t_est = np.zeros((nrow, ncol), dtype='float32')
    count = np.zeros((nrow, ncol), dtype='float32')
    conf = np.zeros((nrow, ncol), dtype='float32')

    [x, y] = get_patch_indices(nrow, ncol, patch_X, patch_Y,
                               stride_X, stride_Y)

    # estimate t in each patch
    for idx in xrange(x.size):
        for idy in xrange(y.size):
            patch = im[x[idx]:x[idx]+patch_X, y[idy]:y[idy]+patch_Y,
                       :].copy()

            var_test = patch_variance_test(patch, var_thr)
            edge_test = patch_edge_test(patch, edge_thr)
            (airlight_angle_test, airlight_cos_dist) \
                = patch_airlight_angle_test(patch, var_thr, angle_thr,
                                            A, angle_test_pixel_frac)

            if airlight_angle_test:
                pass

            if not (var_test and not edge_test and airlight_angle_test):
                continue

            (t_out, nconf) = search_t_binary(patch, comp_model, start_t, end_t,
                                             A, t_tol)

            # computed_conf = nconf*airlight_cos_dist
            computed_conf = nconf

            est_patch = t_est[x[idx]:x[idx]+patch_X,
                              y[idy]:y[idy]+patch_Y].copy()

            c_patch = count[x[idx]:x[idx]+patch_X,
                            y[idy]:y[idy]+patch_Y].copy()
            est_patch += t_out
            c_patch += 1

            t_est[x[idx]:x[idx]+patch_X, y[idy]:y[idy]+patch_Y] = est_patch
            count[x[idx]:x[idx]+patch_X, y[idy]:y[idy]+patch_Y] = c_patch

            conf_patch = conf[x[idx]:x[idx]+patch_X,
                              y[idy]:y[idy]+patch_Y].copy()
            
            conf_patch = np.maximum(conf_patch, computed_conf)
            conf[x[idx]:x[idx]+patch_X, y[idy]:y[idy]+patch_Y] = conf_patch
    
    nzidx = (count != 0)
    t_est[nzidx] = t_est[nzidx]/count[nzidx]

    # Now need to interpolate t
    L = get_laplacian_4neigh(im, longrange=True)
    # L is a sparse matrix not np.ndarray
    # it has zero at places with no estimates, i.e. the discarded patches
    sigma = sparse.dia_matrix((conf.flatten(), [0]),
                              shape=(numpixel, numpixel), dtype='float32')

    t_interp_col = splinalg.spsolve(sigma + alpha * L, sigma*t_est.flatten())
    t_interp = t_interp_col.reshape((nrow, ncol))
    
    t_interp = np.clip(t_interp, 0, 1)
    t_est = np.clip(t_est, 0, 1)

    # recover
    A = A.reshape((1, 1, 3))
    out_im = A + (im - A) / np.tile(t_interp[:, :, np.newaxis], (1, 1, 3))

    out_im = np.clip(out_im, 0, 1)

    return (out_im, t_est, t_interp)


def dehaze_comp2(im, A, comp_model, patch_X, patch_Y, stride_X, stride_Y,
                 var_thr, edge_thr, angle_thr, angle_test_pixel_frac, alpha,
                 start_t, end_t, t_tol):
    '''
    Modifies dehaze_comp to make the inferences in batches for speed-up
    '''

    [nrow, ncol, nch] = im.shape
    
    # preallocate
    t_est = np.zeros((nrow, ncol), dtype='float32')
    count = np.zeros((nrow, ncol), dtype='float32')
    conf = np.zeros((nrow, ncol), dtype='float32')

    [x, y] = get_patch_indices(nrow, ncol, patch_X, patch_Y,
                               stride_X, stride_Y)

    # total number of patches before discarding
    numpatch = x.size*y.size
    im_patches = np.zeros((numpatch, patch_X, patch_Y, nch), dtype='float32')
    patch_used = np.ones((numpatch,), dtype='bool')
    # dump the patches
    sel_patch_idx = 0
    patch_idx = -1
    for idx in xrange(x.size):
        for idy in xrange(y.size):
            patch = im[x[idx]:x[idx]+patch_X, y[idy]:y[idy]+patch_Y,
                       :].copy()
            patch_idx += 1

            # discard patches here
            var_test = patch_variance_test(patch, var_thr)
            edge_test = patch_edge_test(patch, edge_thr)

            # take patch if it passes variance test and fails edge_test
            discard = not var_test or edge_test
            if discard:
                patch_used[patch_idx] = False
                continue
            
            im_patches[sel_patch_idx, :, :, :] = patch
            sel_patch_idx += 1

    patches = im_patches[:sel_patch_idx, :, :, :]
    # find t's only in the extracted patches
    (t_out_patches, nconf_patches) \
        = search_t_binary_batch(patches, comp_model, start_t, end_t,
                                A, t_tol)
    
    # move from patches to image. Aggregate
    sel_patch_idx = 0  # reusing
    patch_idx = -1
    for idx in xrange(x.size):
        for idy in xrange(y.size):
            patch_idx += 1
            if not patch_used[patch_idx]:
                continue
            
            est_patch = t_est[x[idx]:x[idx]+patch_X,
                              y[idy]:y[idy]+patch_Y].copy()

            c_patch = count[x[idx]:x[idx]+patch_X,
                            y[idy]:y[idy]+patch_Y].copy()

            conf_patch = conf[x[idx]:x[idx]+patch_X,
                              y[idy]:y[idy]+patch_Y].copy()

            # update
            est_patch += t_out_patches[sel_patch_idx]
            c_patch += 1
            conf_patch = np.maximum(conf_patch, nconf_patches[sel_patch_idx])
            sel_patch_idx += 1

            t_est[x[idx]:x[idx]+patch_X, y[idy]:y[idy]+patch_Y] = est_patch
            count[x[idx]:x[idx]+patch_X, y[idy]:y[idy]+patch_Y] = c_patch
            conf[x[idx]:x[idx]+patch_X, y[idy]:y[idy]+patch_Y] = conf_patch

    # average the aggregated t
    nzidx = (count != 0)
    t_est[nzidx] = t_est[nzidx]/count[nzidx]
    t_est = np.clip(t_est, 0, 1)
    
    # interpolate t
    t_interp = t_interp_laplacian(t_est, conf, im)

    # recover
    A = A.reshape((1, 1, 3))
    out_im = A + (im - A) / np.tile(t_interp[:, :, np.newaxis], (1, 1, 3))

    out_im = np.clip(out_im, 0, 1)

    return (out_im, t_est, t_interp)

    
def search_t_binary_batch(haze_patches, comparator, start_t, end_t,
                          A, step_tol):
    '''
    Do the binary search of t in a batch
    haze_patches -> (numpatch * patch_X * patch_Y * nch)
    '''

    (numpatch, nrow, ncol, nch) = haze_patches.shape
    
    # assuming start_t < end_t
    t_begin = start_t*np.ones((numpatch,), dtype='float32')
    t_end = end_t*np.ones((numpatch,), dtype='float32')
    conf = np.ones((numpatch,), dtype='float32')

    # A is 1 x 1 x 3
    A_batch = A[np.newaxis, :, :, :]
    
    # abs() is not required as t_end > t_begin. Always.
    while np.all((t_end - t_begin) > step_tol) and np.all(t_end > t_begin):
        t_mid = (t_begin + t_end)/2.0

        # dehaze in batch
        t_mid_batch = np.tile(t_mid[:, np.newaxis, np.newaxis, np.newaxis],
                              (1, nrow, ncol, nch))
        p_d = A_batch + (haze_patches - A_batch)/t_mid_batch
        p_d = np.clip(p_d, 0, 1)

        r = comparator.predict([haze_patches.transpose((0, 3, 2, 1)),
                                p_d.transpose((0, 3, 2, 1))])

        a = r[:, 0]
        b = r[:, 1]

        # need to stop index updates if t_end - t_begin < step_tol
        not_done_indices = (t_end - t_begin) > step_tol
        
        # indices with dehazed one is bad
        dehazed_bad_indices = a > b
        # the dehazed one is bad
        # no need to search below t_mid
        bad_upd_idx = np.logical_and(dehazed_bad_indices, not_done_indices)
        t_begin[bad_upd_idx] = t_mid[bad_upd_idx]

        # the dehazed one is good
        # no need to search above t_mid
        dehazed_good_indices = np.logical_not(dehazed_bad_indices)
        good_upd_idx = np.logical_and(dehazed_good_indices, not_done_indices)
        t_end[good_upd_idx] = t_mid[good_upd_idx]

    # need to handle out of range values and t_end < t_begin case
    conf[t_begin > t_end] = 0

    # this does not have check, as it is done in search_t_binary2
    # whether the t tries to go out of range

    t_mid = (t_begin + t_end)/2.0
    # t_mid = t_end

    return (t_mid, conf)
    

def search_t_binary(haze_patch, comparator, start_t, end_t, A, step_tol):
    '''
    The main difference from method2 is the step size is halved after
    each step and the remaining part is changed accordingly
    '''

    # assuming start_t < end_t
    t_begin = start_t
    t_end = end_t
    conf = 1

    # abs() is not required as t_end > t_begin. Always.
    while (t_end - t_begin) > step_tol and \
          (t_end > t_begin):
        
        t_mid = (t_begin + t_end)/2.0
        p_d = dehaze_patch(haze_patch, t_mid, A)

        [a, b] = comp_patches(comparator, haze_patch, p_d)

        if a > b:
            # the dehazed one is bad
            # no need to search below t_mid
            t_begin = t_mid
        else:
            t_end = t_mid

    # need to handle out of range values and t_end < t_begin case
    if t_begin > t_end:
        conf = 0

    # this does not have check, as it is done in search_t_binary2
    # whether the t tries to go out of range

    t_mid = (t_begin + t_end)/2.0

    return (t_mid, conf)


def comp_patches(comparator, p1, p2):
    '''
    compare the patches using the comparator and return the output
    '''
    r = comparator.predict([p1.transpose()[np.newaxis],
                            p2.transpose()[np.newaxis]])

    a = r[0, 0]
    b = r[0, 1]
    return (a, b)


def t_interp_laplacian(t_est, conf, im):
    '''
    interpolate the t_est
    '''
    [nrow, ncol, nch] = im.shape
    numpixel = nrow * ncol
    
    L = get_laplacian_4neigh(im, longrange=True)
    # L is a sparse matrix not np.ndarray
    # it has zero at places with no estimates, i.e. the discarded patches
    sigma = sparse.dia_matrix((conf.flatten(), [0]),
                              shape=(numpixel, numpixel), dtype='float32')

    # this is faster than spsolve but that is not used in this code
    # from sksparse.cholmod import cholesky  # this is required otherwise
    # chol = cholesky(sigma + alpha * L)
    # t_interp_col = chol.solve_A(sigma*t_est.flatten())

    t_interp_col = splinalg.spsolve(sigma + alpha * L, sigma*t_est.flatten())
    t_interp = t_interp_col.reshape((nrow, ncol))
    
    t_interp = np.clip(t_interp, 0, 1)

    return t_interp


if __name__ == '__main__':
    inp_dir = './haze_image'
    image_files = sorted(os.listdir(inp_dir))
    
    model_file = './model/comp_c_tpartition_30comp_2A.h5'
    out_dir = './out'

    patch_X = 10
    patch_Y = 10
    stride_X = patch_X/2
    stride_Y = patch_Y/2
    
    var_thr = 0.02
    edge_thr = 0.5

    angle_thr = 10
    angle_test_pixel_frac = 0.5
    alpha = 0.1

    start_t = 0
    end_t = 1
    step_tol = 0.0001

    w_op = True

    comp_model = load_model(model_file)

    for sel_idx in xrange(len(image_files)):
        inp_file = image_files[sel_idx]

        sys.stdout.write('[{}/{}] - {} '.format(sel_idx+1, len(image_files),
                                                inp_file))
        sys.stdout.flush()

        in_im = img_as_float(skio.imread(os.path.join(inp_dir, inp_file)))

        tic = time.clock()
        A_comp = compute_A_DCP(in_im)

        im = in_im
        A = A_comp.reshape((1, 1, 3))

        sys.stdout.write(' {} '.format(A))
        (out_im, t_est, t_out) \
            = dehaze_comp2(im, A, comp_model, patch_X, patch_Y,
                           stride_X, stride_Y, var_thr, edge_thr,
                           angle_thr, angle_test_pixel_frac,
                           alpha, start_t, end_t, step_tol)

        toc = time.clock()
        print 'Runtime: {}'.format(toc - tic)

        out_im_final = out_im

        in_file_name = os.path.splitext(inp_file)[0]
        out_file_prefix = os.path.join('.', out_dir, in_file_name)

        if w_op:
            skio.imsave(out_file_prefix+'_out.png', out_im_final)
            skio.imsave(out_file_prefix+'_t.png', t_out)
            skio.imsave(out_file_prefix+'_t_est.png', t_est)
