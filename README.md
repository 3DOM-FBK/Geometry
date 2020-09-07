# Geometry: a library for managing and filtering sparse 3D reconstructions
**Authors**: [Alessandro Torresani](https://3dom.fbk.eu/people/profile/atorresani), [Elisa Mariarosaria Farella](https://3dom.fbk.eu/people/profile/elifarella) and [Fabio Remondino](https://3dom.fbk.eu/people/profile/remondino)

This repository contains a simple and flexibile tool that can load and filter sparse photogrammetric 3D reconstructions, compute, combine and aggregate photogrammetric features (i.e. for removing bad quality 3D tie-points and the corresponding image observations). Results can be re-imported in your favorite 3D reconstruction software.  

### Citation
```
@article{farella2020refining,
  title={Refining the Joint 3D Processing of Terrestrial and UAV Images Using Quality Measures},
  author={Farella, Elisa Maria Rosaria and Torresani, Alessandro and Remondino, Fabio},
  journal={Remote Sensing},
  volume={12},
  number={2873},
  year={2020}
}

Farella, E.M.; Torresani, A.; Remondino, F. Refining the Joint 3D Processing of Terrestrial and UAV Images Using Quality Measures. Remote Sens. 2020, 12, 2873 - https://www.mdpi.com/2072-4292/12/18/2873/htm
```
### Prerequisites
Python3 and Numpy


## How to run
The main steps are:
1) Estimation, in an external software, of the sparse photogrammetric 3D reconstruction. Export the sparse point cloud, camera orientations and intrinsics. 
2) Loading of the 3D reconstruction (step 1) in Geometry and computation of the features (reprojection error, camera viewing angles, etc.).
3) Combination and aggregation of the computed features for editing or filtering the 3D reconstruction.
4) Export of the modified 3D reconstruction for further processing in the favorite photogrammetric software. 

## Example: use Geometry to filter out noisy 3D tie-points
![Figure](docs/graphical_abstract.jpg)
We used this tool in the above pubblication *(Farella et al., 2020)* to identify noisy 3D tie-points and filter them out from the 3D reconstruction. See *(Farella et al., 2020)* for the details.

### Compute the features
```
python3 examples/compute_features.py
```
This scripts computes, for each 3D tie point, the following photogrammetric quality features:
*  mean reprojection error
*  multiplicity (how many images contributed to triangulate the point)
*  maximum intersection angle between the observing images

It takes several arguments:
* ```--input```: path to the 3D reconstruction (so far only [Bundler](https://www.cs.cornell.edu/~snavely/bundler/bundler-v0.4-manual.html) format *".out"* is supported)
* ```--intrinsics```: path to the camera intrinsic file
* ```--intrinsic_format```: format of the intrinsic file (metashape/openCV)
* ```--output```: output folder where the features and the program log will be saved

In *tests/test_dataset* there is a small dataset that you can use to test the commands. In this case the command would be
```bash
python3 examples/compute_features.py --input tests/test_dataset/3D_reconstruction.out --intrinsics tests/intrinsics.txt --intrinsic_format metashape --output .
```
**NOTE 1**: the intrinsics file should contain a line for every image of the dataset. The line order depends on the ordering of the image filenames when sorted in alphabetically crescent order. Each line must be formatted as follows:
```bash
image_id f cx cy k1 k2 k3 p1 p2 img_width img_height
```
See *tests/intrinsics.txt* for an example. In this case, the images were acquired with two different cameras. The images of the first camera were named *DSC_[0-4].png*. The images of the second camera were names *IMG_[0-4].png*. So the instrinsic file should contain first the camera information of the group *DSC_[0-4].png*, and after that of the group *IMG_[0-4].png*.  

The ```intrinsic_format``` is used to specify if the camera center (cx cy) is given as an offset (metashape) or as an absolute value (openCV).


### Filtering 
The input of the filtering script is:
* The features computed in the previous step.
* The variance (*sigma*) of the estimated 3D tie point coordinates. 

The latter is not computed in Geometry and it must be exported from the bundle adjustment statistics of the 3D reconstruction software. For our tests we used a script kindly provided by Agisoft (whom we thank). You can find it in *lib/metashape_save_covariance.py*. This script will add an additional menu in your Metashape toolbar.

To run the filtering script
```bash
python3 examples/filter.py 
```
It takes several arguments: 
* ```--input```: path to the 3D reconstruction (only [Bundler](https://www.cs.cornell.edu/~snavely/bundler/bundler-v0.4-manual.html) format *".out"* is supported now).
* ```--intrinsics```: path to the camera intrinsic file.
* ```--intrinsic_format```: format of the intrinsic file (metashape/openCV)
* ```--features```: path to the file "features.txt" computed in the previous step
* ```--sigma```: path to the file "sigma.txt" exported by *lib/metashape_save_covariance.py*
* ```--output```: output folder where the filtered 3D reconstruction (format .out [Bundler](https://www.cs.cornell.edu/~snavely/bundler/bundler-v0.4-manual.html)) and the program log will be saved. 

Supposing that we use the test dataset and the "*features.txt*" was saved in *tests/test_dataset*, the command would be
```bash
python3 examples/filter.py --input tests/test_dataset/3D_reconstruction.out --intrinsics tests/intrinsics.txt --intrinsic_format --features tests/test_dataset/features.txt --sigma tests/test_dataset/sigma.txt metashape --output .
```

## Note on software compatibility
Currently the library supports only 3D reconstructions in [Bundler](https://www.cs.cornell.edu/~snavely/bundler/bundler-v0.4-manual.html) format. This applies both to the import and the export parts. In the presented example and in the pubblication *(Farella et al., 2020)*, we used Agisoft Metashape (version 1.6.3). The library, however, should theoretically work with any 3D reconstruction software that supports [Bundler](https://www.cs.cornell.edu/~snavely/bundler/bundler-v0.4-manual.html) import/export. 
