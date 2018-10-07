import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt

# Prepare data

def normalize_data(data):
    return (data - data.mean()) / data.std()



num_house = 200
np.random.seed(42)

house_size = np.random.randint(low=1000, high=3500, size=num_house)
house_price = house_size*100.0 + np.random.randint(low=20000,high=70000,size=num_house)

plt.plot(house_size,house_price,"bx")
plt.ylabel("Price")
plt.xlabel("Size")
plt.show()


training_set_size = math.floor(num_house*0.7)


training_house_size = np.asarray(house_size[:training_set_size])
training_house_price = np.asarray(house_price[:training_set_size])

test_house_size = np.asarray(house_size[training_set_size:])
test_house_price = np.asarray(house_price[training_set_size:])

training_house_size_norm = normalize_data(training_house_size)
training_house_price_norm = normalize_data(training_house_price)

test_house_size_norm = normalize_data(test_house_size)
test_house_price_norm = normalize_data(test_house_price)

# Inference


tf_house_price = tf.placeholder('float', name='house_price')
tf_house_size = tf.placeholder('float',name='house_size')

tf_size_factor_param = tf.Variable(initial_value=np.random.randn(),name='size_factor')
tf_price_offset_param = tf.Variable(initial_value=np.random.randn(), name='price_offset')

tf_price_predict = tf.add(tf.multiply(tf_house_size,tf_size_factor_param),tf_price_offset_param)

# Loss

tf_cost = tf.reduce_sum(tf.pow(tf_price_predict - tf_house_price, 2)) / (2*training_set_size)
learning_rate = 0.04

# Optimize

tf_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss=tf_cost)

# Execution

tf_init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(tf_init)

    display_every = 2
    train_iters = 50

    for iter in range(train_iters):

        for (X,y) in zip(training_house_size_norm, training_house_price_norm):
            sess.run(tf_optimizer, feed_dict = {tf_house_size:X, tf_house_price: y})

            if (iter+1) % 2 == 0:
                c = sess.run(tf_cost, feed_dict = {tf_house_size: training_house_size_norm, tf_house_price: training_house_price_norm})
                print("iteration #:", '%04d' % (iter + 1), "cost=", "{:.9f}".format(c), \
                "size_factor=", sess.run(tf_size_factor_param), "price_offset=", sess.run(tf_price_offset_param))