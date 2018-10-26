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

# Generate data from clustered patches
# generate them by taking random t and A's
# add data for comparison with only haze patches

import numpy as np
import sys
from scipy.spatial.distance import pdist, cdist
import math

from lib import en_haze, dehaze_patch


def add_patch(better_patch, worse_patch, patches_array, label_array, p_idx):
    '''
    Add the provided patches to preallocated array
    corresponding label also needs to be generated
    '''
    if np.random.random() >= 0.5:
        patches_array[0, p_idx, :, :, :] = better_patch.transpose()
        patches_array[1, p_idx, :, :, :] = worse_patch.transpose()
        label_array[p_idx, :] = [1, 0]
    else:
        patches_array[1, p_idx, :, :, :] = better_patch.transpose()
        patches_array[0, p_idx, :, :, :] = worse_patch.transpose()
        label_array[p_idx, :] = [0, 1]


if __name__ == '__main__':
    patch_X = 10
    patch_Y = 10
    patch_ch = 3
    
    cluster_center_f = './data/clustered_patches.npy'
    file_prefix = 'p_c_tpartition_30comp_2A'
    outf_patches = file_prefix+'.npy'
    outf_labels = file_prefix+'_labels.npy'
    outf_t = file_prefix+'_ts.npy'
    outf_A = file_prefix+'_As.npy'

    patch_len = patch_X*patch_Y*patch_ch
    # valid t range
    t_max = 1
    t_min = 0
    # t generation range
    t_g_max = 0.9
    t_g_min = 0.1
    t_step = 0.0005  # the difference between 2 t's
    t_gen_std_dev = 0.15
    airlight_angle_thresh = 2.5
    airlight_angle_frac = 0.3

    print "Loading data"
    # this 'centers' is of size (nclusters x nfeatures)
    # out of the features first 'patch_len' values are RGB values of the patch,
    # the remaining ones are its X and Y gradient with borders removed
    centers = np.load(cluster_center_f)[:, :patch_len]
    numpatch = centers.shape[0]
    print 'Done'

    # number of different t's generated to compare with one hazed patch
    n_comp = 30
    n_A = 2
    n_out = numpatch * n_comp * n_A  # total number of output patches
    # allocate the data
    X_all = np.zeros((2, n_out, patch_ch, patch_Y, patch_X),
                     dtype='float32')
    Y_all = np.zeros((n_out, 2), dtype='float32')
    t_all = np.zeros((numpatch, n_A, n_comp+1), dtype='float32')
    A_all = np.zeros((n_out, 1, 1, patch_ch), dtype='float32')

    # paramters for partition
    n_partition = 5
    # sample size
    n_good = n_comp / 2
    n_bad = n_comp - n_good  # if n_comp is not a multiple of 2
    n_in_good_p = np.ones(n_good, dtype='int32')*(n_good / n_partition)
    n_in_good_p[0] += n_good % n_partition
    n_in_bad_p = np.ones(n_bad, dtype='int32')*(n_bad / n_partition)
    n_in_bad_p[-1] += n_bad % n_partition
    partition_frac = 0.5 ** np.r_[1:n_partition]

    X_idx = 0
    for i in xrange(numpatch):
        # this is a clean patch
        patch = centers[i, :].reshape((patch_X, patch_Y, patch_ch))

        # Show status
        sys.stdout.write('\r')
        sys.stdout.write('({}/{}) - {}%'.format(i, numpatch, i*100/numpatch))
        sys.stdout.flush()

        while True:
            t_g = (t_g_max - t_g_min)*np.random.random() + t_g_min

            # is this necessary at all ?
            # as t_g_max < t_max and t_g_min > t_min
            if (t_g >= t_min) and (t_max >= t_g):
                # t is in valid range use it
                break

        # debug
        sys.stdout.write(' t_g = {}'.format(t_g))
        
        for j in xrange(n_A):
            # generate one `A'
            while True:
                A = np.random.random((1, 1, 3))
                # should this be allowed ? any value ?
                # let's keep it like this for now
                haze_patch = en_haze(patch, t_g, A)

                # what if A is similar to the patch ?
                # how to check ?
                # angular distance. but pixel to single value?
                # what to do when they are similar
                # can't be done, generate another A
                # may consider only the smooth patches (not done now)
                # cosine distance is NaN when a pixel is [0, 0, 0]

                x_a = patch.reshape((patch_X*patch_Y, -1))
                x_b = A.reshape((1, patch_ch))
                d = cdist(x_a, x_b, metric='cosine')

                thr_test = d > (1 - math.cos(math.radians(
                    airlight_angle_thresh)))

                test_val = np.sum(thr_test) > ((
                    patch_X*patch_Y)*airlight_angle_frac)

                if test_val:
                    break

            # generate t's using variable length partition of t range
            # preallocate
            # can be moved outside but will re-initialize things
            good_t = np.zeros(n_good, dtype='float32')
            bad_t = np.zeros(n_bad, dtype='float32')

            good_r = (t_g + t_step, t_max - t_step)
            bad_r = (t_min, t_g - t_step)

            good_p = good_r[0] + np.cumsum((good_r[1] - good_r[0])
                                           * partition_frac[::-1])
            bad_p = bad_r[0] + np.cumsum((bad_r[1] - bad_r[0])*partition_frac)
            # need to add boundary to these and generate unifrom random

            good_t_r = np.append(np.insert(good_p, 0, good_r[0]), good_r[1])
            for idx in xrange(good_t_r.size - 1):
                l = good_t_r[idx]
                h = good_t_r[idx + 1]

                while True:
                    t_comp = (h - l) * np.random.random(n_in_good_p[idx]) + l

                    # range checking is not required
                    # the t_comp's should not be too close
                    dist = pdist(np.reshape(t_comp, (-1, 1)),
                                 metric='minkowski', p=1)
                    if np.any(dist < t_step):
                        continue

                    in_index = idx*n_in_good_p[idx]
                    good_t[in_index:in_index+n_in_good_p[idx]] = t_comp
                    break

            # print good_t
            assert(np.all(good_t > t_g))

            # similar for bad t's
            bad_t_r = np.append(np.insert(bad_p, 0, bad_r[0]), bad_r[1])
            for idx in xrange(bad_t_r.size - 1):
                l = bad_t_r[idx]
                h = bad_t_r[idx + 1]

                while True:
                    t_comp = (h - l) * np.random.random(n_in_bad_p[idx]) + l

                    # range checking is not required
                    # gap checking ?
                    # the t_comp's should not be too close
                    dist = pdist(np.reshape(t_comp, (-1, 1)),
                                 metric='minkowski', p=1)
                    if np.any(dist < t_step):
                        continue

                    in_index = idx*n_in_bad_p[idx]
                    bad_t[in_index:in_index+n_in_bad_p[idx]] = t_comp
                    break

            # print bad_t
            assert(np.all(bad_t < t_g))
                
            for t_good in good_t:
                d_p = dehaze_patch(haze_patch, t_good, A)
                add_patch(d_p, haze_patch, X_all, Y_all, X_idx)
                A_all[X_idx, :, :, :] = A
                X_idx += 1

                assert(np.sum((haze_patch - d_p)**2) != 0)

            for t_bad in bad_t:
                d_p = dehaze_patch(haze_patch, t_bad, A)
                add_patch(haze_patch, d_p, X_all, Y_all, X_idx)
                A_all[X_idx, :, :, :] = A
                X_idx += 1

                assert(np.sum((haze_patch - d_p)**2) != 0)

            t_all[i, j, 0] = t_g
            t_all[i, j, 1:n_good+1] = good_t
            t_all[i, j, n_good+1:] = bad_t
            
    # all the patches are added
    # save them along with the labels
    # save only the amount utilized
    np.save(outf_patches, X_all[:, :X_idx])
    np.save(outf_labels, Y_all[:X_idx])
    # np.save(outf_t, t_all)
    # np.save(outf_A, A_all[:X_idx])
