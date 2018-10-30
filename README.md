# Code of the paper "Learning a Patch Quality Comparator for Single Image Dehazing"
[Project Page](https://github.com/san-santra/dehaze_t_comparator)

| Input        | Dehzed         | 
|:-------------:|:-------------:|
| ![input](http://san-santra.github.io/public/haze_image/2230089563_06d4982122_z.jpg)| ![dehazed](http://san-santra.github.io/comp_t18/results/2230089563_06d4982122_z_out.jpg) |

## Dependency
* For Running
    * Python 2
    * keras (with any backend)
    * scikit-image
    * scipy
    * scikit-sparse
    * numpy

Both *scipy* and *scikit-sparse* are used for sparse matrix computation. The *scikit-sparse* has been used for solving linear equation with sparse matrix. Although the same thing can be achieved using *scikit* only, *scikit-sparse* is faster as it uses Cholesky decomposition using CHOLMOD library. 

* Additional dependency for training
    *   opencv (required for clustering)

## Running
Running `python dehaze_im_bin_search.py` dehazes all the images present in `haze_image` folder and stores the output in `out` folder. 

## Files
```
.
├── cluster_data.py                             # extract patches from images and runs k-means
├── data                                        # training data. need to download separately
│   ├── README.md                               # the details are given in this README.md
│   └── training_images
│       └── filelist.txt                        # files used for clustering
├── data_gen.py                                 # comparator training data generator
├── dehaze_im_binsearch.py                      # dehaze image with a trained comparator
├── haze_image                                  # hazy images
│   └── 2230089563_06d4982122_z.jpg
├── lib                                         # helper functions
│   ├── __init__.py
│   ├── lib.py
├── LICENSE
├── model                                       # trained comparator model
│   └── comp_c_tpartition_30comp_2A.h5
├── out                                         # output obtained from the hazy images
│   ├── 2230089563_06d4982122_z_out.png         # dehazed output
│   ├── 2230089563_06d4982122_z_t_est.png       # estimated transmittance before smoothing
│   └── 2230089563_06d4982122_z_t.png           # smoothed and interpolated transmittance
├── README.md
├── train_comp_model.py                         # trains the comparator with the generated data
├── visualize_cluster_centers.py                # for visualizing the generated cluster centers
```

For dehazing an image running the `dehaze_im_binsearch.py` is sufficient. To train a new comparator the following steps need to be followed.
1. Some fog-free images needs to be gathered. We have used the fog-free files given by Choi et al [1]. The details can be found in `data/README.md`.
2. Then running `cluster_data.py` will extract some patches and cluster them.
3. Now to generate the training data for the comparator, `data_gen.py` needs to be run. 
4. After this calling `train_comp_model.py` trains the comparator with the generated data.

## Publication
Sanchayan Santra, Ranjan Mondal, and Bhabatosh Chanda. "Learning a Patch Quality Comparator for Single Image Dehazing." IEEE Transactions on Image Processing 27, no. 9 (2018).

## Reference
1. L. K. Choi, J. You, and A. C. Bovik, "LIVE Image Defogging Database," Online: http://live.ece.utexas.edu/research/fog/fade_defade.html, 2015. 
