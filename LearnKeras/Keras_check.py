from keras import backend as kbe
import os
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

data = kbe.variable(np.random.random((4,2)))

zero = kbe.zeros_like(data)
print(kbe.eval(zero))