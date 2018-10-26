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

# open patches file saved by the clustering step and generate one
# big image for visalization

import numpy as np
import skimage.io as skio

if __name__ == '__main__':
    patch_X = 10
    patch_Y = 10
    patch_ch = 3

    patches_file = 'clustered_patches.npy'
    outf_im = 'patches.png'

    patches = np.load(patches_file)
    numpatch = patches.shape[0]
    patch_len = patch_X*patch_Y*patch_ch

    # allocate the output image
    w = int(np.ceil(np.sqrt(numpatch)))
    h = int(np.ceil(numpatch / float(w)))
    out_w = w*patch_X
    out_h = h*patch_Y
    out_im = np.zeros((out_h, out_w, patch_ch), dtype='float32')
    for i in xrange(numpatch):
        idx = (i / w)*patch_X
        idy = (i % w)*patch_Y
        print idx, idy
        p = patches[i, :patch_len].reshape((patch_X, patch_Y, patch_ch))
        print out_im[idx:idx+patch_X, idy:idy+patch_Y, :].shape
        out_im[idx:idx+patch_X, idy:idy+patch_Y, :] = p
    
    skio.imsave(outf_im, out_im)
