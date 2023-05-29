USAGE GUIDE:<br />
Library provides following functions<br />

parameters common for many functions:<br />
  -image is a 2d numpy array consisting of 1 and 0 that will be transformed<br />
  -transform_matrix is a numpy array of size 2 x 2, positive-definite symmetric matrix that defines metric<br />
  -is_signed is a flag if we need signed MDT (0 - MDT, 1 - signed MDT)<br />
  -connectivity_type is a string equal to 8-connectivity or 4-connectivity that defines how we construct image graph<br />
  
mahalanobis_transformation.MDT_connectivity(image, transform_matrix, connectivity_type, is_signed)<br />
mahalanobis_transformation.MDT_brute(image, transform_matrux, is_signed)<br />
mahalanobis_transformation.MDT_window(image, transform_matrix, distance)<br />
  -distance is a real number that defines the neighbouhood size (pixels located further than distance from current will be ignored)<br />
mahalabonis_transformation.MDT_ellipse(image, lambda1, lambda2, theta, connectivity_type, is_signed)<br />
  -lambda1, lambda2, theta are ellipse params <br />
![image](https://github.com/StalkerShurik/MahalanobisDistanceTransformation/assets/67455670/7f8e33b1-dc4e-4775-ba1b-ee35d0fdcd3c)<br />
  
all functions return numpy arrays (tranformed images) <br />
