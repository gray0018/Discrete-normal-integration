# Discrete normal integration

## What is the Poisson normal integration?
Poisson normal integration is an efficient method to reconstruct surfaces from given normal maps. by solving a Poisson Equation. The Equation comes from solving a functional. We can also add sparse depth prior to refine the result.<br>

Find the depth funciton <img src="http://latex.codecogs.com/gif.latex?z"/>
whose gradient best approximates the surface normal (represented
by <img src="http://latex.codecogs.com/gif.latex?p, q"/>). 


Equivalent to minimizing 
<img src="http://latex.codecogs.com/gif.latex?J(z)"/>:

<p align="center">
<img src="http://latex.codecogs.com/gif.latex?J%28z%29%3D%5Ciint%28%28z_x-p%29%5E2&plus;%28z_y-q%29%5E2%29dxdy"/>
</p>

Euler-Lagrange equation for this functional is a Poisson equation:

<p align="center">
<img src="http://latex.codecogs.com/gif.latex?%5CDelta%20z%3Dp_x&plus;q_y"/>
</p>

<p align="center">
<img src="data/surface2depth.png" width="668">
</p>

## How to use?
Clone the repository and run the main.py:
```
python main.py filename_of_normal_map
```
We have already prepared normal maps for testing. They are put under /data. You can run:
```
python main.py data/scholar.png
```
If you want to add depth prior, you can try:
```
python main.py data/bunny/bunny_normal.npy -d data/bunny/bunny_sparse_depth.npy
```
For more information:
```
python main.py -h
```


- As for the normal maps' formats, we support all the formats which could be read by OpenCV(e.g. png, jpg, bmp) and also npy file in numpy.<br>
**Notice that the background of the normal map should be filled (0, 0, 0)**.<br>
- For adding sparse depth prior, **keep the resolution of depth map same with the normal map**. For unused pixels in depth map, **set them into NaN**. And **use npy format for depth map** to ensure the accuracy<br>


## Result

### Without adding depth prior
Normal map             |  Reconstructed surface 
:-------------------------:|:-------------------------:
<img src="data/scholar.png" width="512px">  |  <img src="data/scholar.gif" width="512px">

### Adding depth prior
- We use the normal map of standord bunny as input, and add sparse depth prior from different parts.
<img src="data/normal_depth_prior.svg" >

- Below are the mean absolute error maps of not adding depth prior and adding depth prior. We can see that adding depth prior significantly reduces the depth error.
<img src="data/with_and_wo_depth_prior.svg" >


## Dependencies
- OpenCV-Python
- NumPy
- SciPy
- scikit-learn
