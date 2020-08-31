# Geometry: an independent tool for managing sparse photogrammetric reconstructions
**Authors**: [Alessandro Torresani](https://3dom.fbk.eu/people/profile/atorresani), [Elisa Farella](https://3dom.fbk.eu/people/profile/elifarella) and [Fabio Remondino](https://3dom.fbk.eu/people/profile/remondino)

This repository contains a simple and flexibile tool that can load sparse photogrammetric 3D reconstructions, compute geometric or photogrammetric features, aggregate those features in equations and apply them on the 3D reconstruction (i.e. for removing bad quality 3D tie-points and the corresponding image observations), and output results that can be re-imported in your favorite 3D reconstruction software.  
<!--- IMAGES HERE-->

### Related publications
<!--- JOURNAL HERE-->

### Prerequisites
Python3 and Numpy


## How to run
The main steps are:
1) Estimation, in an external software, of the sparse photogrammetric 3D reconstruction. Export point cloud, camera orientations and intrinsics. 
2) Loading of the 3D reconstruction in Geometry and computation of the features (reprojection error, camera viewing angles, etc.)
3) Aggregation of the computed features in equations and application of the equations on the 3D reconstruction.
4) Export of the modified 3D reconstruction, ready to be imported back to your 3D reconstruction software.

## Example: use Geometry to filter out noisy 3D tie-points
We used this tool in the above pubblication *(Farella et al., 2020)* to identify noisy 3D tie-points and filter them out from the 3D reconstruction. See *(Farella et al., 2020)* for the details.

### Compute the photogrammetric features
```
python3 examples/compute_features.py
```
This scripts takes several arguments:
* ```--input```: path to the 3D reconstruction (only [Bundler](https://www.cs.cornell.edu/~snavely/bundler/bundler-v0.4-manual.html) format *".out"* is supported now).
* ```--intrinsics```: path to the camera intrinsic file.
* ```--intrinsic_format```: format of the intrinsic file (metashape/openCV)
* ```--output```: output folder where the features and the program log will be saved

In *tests/test_dataset* there is a small dataset that you can use to test the commands. In this case the command would be
```
python3 examples/compute_features.py --input tests/test_dataset/3D_reconstruction.out --intrinsics tests/intrinsics.txt --intrinsic_format metashape --output .
```
**NOTE**: the intrinsics file should contain a line for every image of the dataset. The line order must respect the order of the images when sorted (filename) in alphabetically crescent order. Each line must be formatted as follows:
```
image_id f cx cy k1 k2 k3 p1 p2 img_width img_height
```
See *tests/intrinsics.txt* for an example. In this case, for example, we have two different cameras (images 0 to 4, images 5 to 9). The intrinsic format is used to specify if the camera center (cx cy) is given as an offset (metashape) or as an absolute value (openCV).

### Filtering 

## Note on software compatibility