# MahalanobisDistanceTransformation
The Mahalanobis distance transform (MDT) maps every image pixel into the smallest distance to regions of interest using wide class of metrics that can be defined with symmetric positive-definite matrices. 
So MDT is a generalization of the Euclidean distance transform that use standard euclidean metric. There are lots of applications of 
transforms like this. For instance, separation of overlapping objects, robot navigation, computer vision tasks 
(for instance image classification in geology or medicine) and many others. The problem is how to compute MDT effectively. 
Using various improvement algorithms it is possible to compute MDT faster than brute force algorithm.  <br />

Compiling guide: <br />
git clone https://github.com/pybind/pybind11 to the root <br /> 
cd build <br />
cmake ../ <br />
make <br />

now you have mahalanobis_transformation.cpython.so file that can be included into Python code
