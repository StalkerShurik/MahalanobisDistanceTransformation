import blob_generator
import mahalanobis_transformation
import numpy as np
from PIL import Image
import time

trans = np.array([[0.5, 0], [0, 20]])
test = blob_generator.generate_samples(5)

def test_MDT_connectivity():
    start = time.time()
    for sample in test:
        transformed_image = mahalanobis_transformation.MDT_connectivity(sample[0], trans, "8-connectivity", 0)
    end = time.time() - start
    print("MDT connectivity works: " + str(end) + " seconds")

def test_MDT_brute():
    start = time.time()
    for sample in test:
        transformed_image = np.abs(mahalanobis_transformation.MDT_brute(sample[0], trans, 0))
    end = time.time() - start
    print("MDT brute works: " + str(end) + " seconds")

def test_MDT_window():
    start = time.time()
    for sample in test:
        transformed_image = np.abs(mahalanobis_transformation.MDT_window(sample[0], trans, 10))
    end = time.time() - start
    print("MDT window works: " + str(end) + " seconds")

test_MDT_connectivity()
test_MDT_brute()
test_MDT_window()


