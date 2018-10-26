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

# Some helper functions

import numpy as np
from scipy import sparse, ndimage
from scipy.spatial import distance
from skimage.color import rgb2gray
from skimage import morphology


def get_laplacian_4neigh(im, longrange=False):
    '''
    computes the weighted laplacian matrix of the given image
    im used for the dimension of the Laplacian Matrix and weights of the edges
    '''

    # parameters
    min_i_diff_sq = 0.0001
    big_window_frac = 0.15
    big_window_overlap_frac = 0.95
    long_range_i_thr = 0.1
    sampling_skip = 3
    nsample = 5

    [nrow, ncol, nch] = im.shape
    numnode = nrow * ncol
    ind = np.r_[:numnode]
    ind_mat = ind.reshape((nrow, ncol))
    # copy only if they are manipulated independently

    # numnode * nch
    im_r = im.reshape((-1, nch))

    # first compute the adjacency matrix
    adjmat = sparse.csr_matrix((numnode, numnode), dtype='float32')

    # here the arrays are row major
    # right edges
    right_neigh_ind = ind_mat + 1
    right_neigh_excl = right_neigh_ind[:, :-1]
    ind_mat_excl = ind_mat[:, :-1]

    # want || I(x) = I(y) ||^2
    neigh_i_diff = im_r[ind_mat_excl.flatten(), :] \
        - im_r[right_neigh_excl.flatten(), :]

    i_d_norm_sq = np.sum(neigh_i_diff*neigh_i_diff, axis=1)
    right_wt = 1 / np.maximum(i_d_norm_sq, min_i_diff_sq)

    right_edges = sparse.coo_matrix((right_wt, (ind_mat_excl.flatten(),
                                                right_neigh_excl.flatten())),
                                    shape=(numnode, numnode)).tocsr()

    right_edges = right_edges.tocsr()
    # add right and left edges
    adjmat = adjmat + right_edges + right_edges.transpose()

    # down edges
    down_neigh_ind = ind_mat + ncol
    down_neigh_excl = down_neigh_ind[:-1, :]
    ind_mat_excl = ind_mat[:-1, :]

    neigh_i_diff = im_r[ind_mat_excl.flatten(), :] \
        - im_r[down_neigh_excl.flatten(), :]

    i_d_norm_sq = np.sum(neigh_i_diff*neigh_i_diff, axis=1)
    down_wt = 1 / np.maximum(i_d_norm_sq, min_i_diff_sq)

    down_edges = sparse.coo_matrix((down_wt, (ind_mat_excl.flatten(),
                                              down_neigh_excl.flatten())),
                                   shape=(numnode, numnode)).tocsr()

    down_edges = down_edges.tocsr()
    # add down and up edges
    adjmat = adjmat + down_edges + down_edges.transpose()

    if longrange:
        # long range edges
        # within a big window, sample pixels
        # add edges between them if they have similar intensity
        big_window = int(np.floor(min(nrow, ncol)*big_window_frac))
        big_window_shift = int(np.ceil(big_window*(1-big_window_overlap_frac)))

        x = np.r_[:nrow - big_window:big_window_shift]
        y = np.r_[:ncol - big_window:big_window_shift]

        l_data = []
        l_row_ind = []
        l_col_ind = []
        # take a big patch -> similar pixels are supposed to be near
        # sample some pixels
        # connect if they have "similar" intensity
        for ridx in xrange(x.size):
            for cidx in xrange(y.size):
                ind_arr_local = ind_mat[x[ridx]:x[ridx]+big_window,
                                        y[cidx]:y[cidx]+big_window].copy()

                candidate_samples_mask = np.ones((big_window, big_window),
                                                 dtype='bool')

                mid_idx = ind_arr_local[0, 0] + \
                    np.ravel_multi_index((big_window/2, big_window/2),
                                         (big_window, big_window))
                mid_p = im_r[mid_idx, :]  # the mid pixel in the window

                # remove 4 neighbors from the candidate
                mid_r = big_window/2
                mid_c = big_window/2
                candidate_samples_mask[mid_r, mid_c-1:mid_c+2] = False
                candidate_samples_mask[mid_r-1:mid_r+2, mid_c] = False
                # a bit of dirty hack
                
                # the samples are taken in every fourth pixel in each axis
                # don't conisder others when sampling
                sample_idx = np.zeros((big_window, big_window), dtype='bool')
                sample_idx[::sampling_skip, ::sampling_skip] = True
                # from these positions samples are to be taken
                candidate_samples_mask[np.logical_not(sample_idx)] = False
                
                candidate_idx = ind_arr_local[candidate_samples_mask]
                np.random.shuffle(candidate_idx)
                candidate_pixels = im_r[candidate_idx[:nsample], :]
                
                # edges are added between the center pixel and one of the pixels
                # sampled from the window if the distance is within a threshold
                
                # requires m_A by n and m_B by n array; n -> no. dimension
                # returns m_A by m_B array as output
                d = distance.cdist(mid_p[np.newaxis], candidate_pixels,
                                   metric='euclidean')
                # mid_p is (3,) so newaxis is required
                sel_p_idx = np.nonzero(d < long_range_i_thr)
                # the returned value is a tuple
                # d is (1, #candidate pixels)
                # so sel_p_idx is (array([0]), nz idxs)
                sel_p_idx = sel_p_idx[1]
                
                if sel_p_idx.size != 0:
                    # if at all there exists a pixel, add the connection
                    # how?
                    l_data.append(1/np.maximum(d[0, sel_p_idx[0]]**2,
                                               min_i_diff_sq))

                    # connect this selected pixel with the center pixel
                    r = candidate_idx[sel_p_idx[0]]
                    c = mid_idx
                    
                    l_row_ind.append(r)
                    l_col_ind.append(c)

        # long range connections
        lr_edge = sparse.coo_matrix((l_data, (l_row_ind, l_col_ind)),
                                    shape=(numnode, numnode))
        print "long range edges: {}".format(lr_edge.nnz)
        lr_edge = lr_edge.tocsr()
        adjmat = adjmat + lr_edge + lr_edge.transpose()

        # if of longrange ends
    
    # So, adjacency matrix done
    degree = adjmat.sum(axis=1)
    degree_mat = sparse.dia_matrix((degree.flatten(), [0]),
                                   shape=(numnode, numnode))

    laplacian = degree_mat - adjmat

    return laplacian


def en_haze(patch, t, A):
    '''
    add haze to patch based on t and A
    '''
    r = patch*t + (1 - t)*A
    r = np.clip(r, 0, 1)  # this assumes float image data

    return r


def dehaze_patch(patch, t, A):
    '''
    Dehaze the given patch
    '''
    j = A + (patch - A)/t
    j = np.clip(j, 0, 1)

    return j


def compute_A_DCP(im):
    '''
    Compute 'A' as described in DCP (He et al.)
    '''
    # Parameters
    erosion_window = 15
    n_bins = 100
    
    # compute the dark channel
    dark = morphology.erosion(np.min(im, 2),
                              morphology.square(erosion_window))
    # max_d = np.max(dark)
    # mask = dark >= 0.99*max_d
    # this may not be the top 1%
    # should ideally done using a histogram
    [h, edges] = np.histogram(dark, n_bins)
    numpixel = im.shape[0]*im.shape[1]
    thr_frac = numpixel*0.99
    csum = np.cumsum(h)
    nz_idx = np.nonzero(csum > thr_frac)[0][0]
    dc_thr = edges[nz_idx]
    mask = dark >= dc_thr
    
    # brightest intensity pixel in the mask
    gray = rgb2gray(im)
    pos_f = np.argmax(gray*mask)
    pos = np.unravel_index(pos_f, gray.shape)

    A = im[pos[0], pos[1], :]

    return A


def compute_A_Tang(im):
    '''
    Compute 'A' as described by Tang et al. (CVPR 2014)
    '''
    # Parameters
    erosion_window = 15
    n_bins = 100

    R = im[:, :, 0]
    G = im[:, :, 1]
    B = im[:, :, 2]
    
    # compute the dark channel
    dark = morphology.erosion(np.min(im, 2),
                              morphology.square(erosion_window))

    [h, edges] = np.histogram(dark, n_bins)
    numpixel = im.shape[0]*im.shape[1]
    thr_frac = numpixel*0.99
    csum = np.cumsum(h)
    nz_idx = np.nonzero(csum > thr_frac)[0][0]
    dc_thr = edges[nz_idx]
    mask = dark >= dc_thr
    # similar to DCP till this step
    # next, median of these top 0.1% pixels
    # median of the RGB values of the pixels in the mask
    rs = R[mask]
    gs = G[mask]
    bs = B[mask]

    Ar = np.median(rs)
    Ag = np.median(gs)
    Ab = np.median(bs)

    return np.dstack((Ar, Ag, Ab))


def discard_patch(patch, var_thr, edge_thr):
    '''
    Check whether to take or discard the patch based on:
    a. variance
    b. existance of edge
    '''
    gray_patch = rgb2gray(patch)
    var = np.std(gray_patch)
    if var < var_thr:
        # low variance means smooth patch
        # discard it
        return True
    else:
        # edge = filters.sobel(gray_patch)
        sx = ndimage.sobel(gray_patch, axis=0, mode='constant')
        sy = ndimage.sobel(gray_patch, axis=1, mode='constant')
        edge = np.hypot(sx, sy)[1:-1, 1:-1]  # ignore borders
        if np.any(edge > edge_thr):
            # contains edge - discard it
            return True

    return False


def patch_variance_test(patch, var_thr):
    '''
    Check whether the standard deviation of the given patch is
    greater than var_thr
    '''

    gray_patch = rgb2gray(patch)
    var = np.std(gray_patch)

    return var > var_thr


def patch_edge_test(patch, edge_thr):
    '''
    Check whether an edge pixel exists in the patch or not
    Existance of edge pixel is checked with canny
    '''

    gray_patch = rgb2gray(patch)
    sx = ndimage.sobel(gray_patch, axis=0, mode='constant')
    sy = ndimage.sobel(gray_patch, axis=1, mode='constant')
    edge = np.hypot(sx, sy)[1:-1, 1:-1]  # ignore borders

    # this np.any is not very good
    return np.any(edge > edge_thr)


def patch_airlight_angle_test(patch, var_thr, angle_thr, A, pixel_frac):
    '''
    Check whether the pixels make small angle with `A'
    A = airlight vector
    Returns: (test_val, cos_dist)
    test_val = true if pixel_frac amount of the pixels are above angle_thr
    cos_dist = cosine distance (1 - cos(x)) of each pixel from `A' with
    same number of rows and columns as patch
    '''

    [nrow, ncol, nch] = patch.shape

    if patch_variance_test(patch, var_thr):
        return (True, np.ones((nrow, ncol)))

    from scipy.spatial.distance import cdist
    import math
    
    x_a = patch.reshape((nrow*ncol, -1))
    x_b = A.reshape((1, nch))
    d = cdist(x_a, x_b, metric='cosine')

    thr_test = d > (1 - math.cos(math.radians(angle_thr)))

    test_val = np.sum(thr_test) > ((nrow*ncol)*pixel_frac)
    cos_dist = d.reshape((nrow, ncol))

    return (test_val, cos_dist)


def get_patch_indices(nrow, ncol, patch_X, patch_Y, stride_X, stride_Y):
    '''
    return indices of the positions to take patch from
    '''

    x = np.r_[:nrow - patch_X:stride_X]
    y = np.r_[:ncol - patch_Y:stride_Y]

    return (x, y)
