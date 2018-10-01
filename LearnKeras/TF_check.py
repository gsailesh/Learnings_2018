# Verify that Tensorflow is working!
import tensorflow as tf
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

# VERSION
print("Tensorflow version is: " + str(tf.__version__))
hello = tf.constant("Hello Tensorflow!")
session = tf.Session()

print(session.run(hello))