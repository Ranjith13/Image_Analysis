#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 13:35:00 2019

@author: ranju
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#loading mnist dataset from tf
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

#initialising nurons or nodes
n_node_hl1 = 500
n_node_hl2 = 500
n_node_hl3 = 500

#initialising classes
n_classes = 10
batch_size = 100

#declaring x and y which is independent and dependent variable
x = tf.placeholder('float')
y = tf.placeholder('float')

#creating neural network model
def neural_network_model(data):
    """
    creating the neural network model
    """
    hidden_1_layer = {'weights':tf.Variable(tf.random.normal([784, n_node_hl1])),
                      'biases':tf.Variable(tf.random.normal([n_node_hl1]))}

    hidden_2_layer = {'weights':tf.Variable(tf.random.normal([n_node_hl1, n_node_hl2])),
                      'biases':tf.Variable(tf.random.normal([n_node_hl2]))}

    hidden_3_layer = {'weights':tf.Variable(tf.random.normal([n_node_hl2, n_node_hl3])),
                      'biases':tf.Variable(tf.random.normal([n_node_hl3]))}

    output_layer = {'weights':tf.Variable(tf.random.normal([n_node_hl3, n_classes])),
                    'biases':tf.Variable(tf.random.normal([n_classes]))}

    # (input * weights) + biasesa
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1) # activation function which is simoid to fier the neurons "rectified linear(relu)"

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

    return output


def train_model(x):
    """
    training the model
    """
    prediction = neural_network_model(x)
    #calculating cost for prediction and y
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = y))

    #optimizer has a parameter which is learning_rate with default value 0.001
    optimizer = tf.train.AdamOptimizer().minimize(cost)#to minimize the cost

    num_of_epochs = 10 #iteration of feedforward and back propogation

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epochs in range(num_of_epochs):
            epochs_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                x_epochs, y_epochs = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict = {x:x_epochs, y:y_epochs})
                epochs_loss += c
            print('Epochs', epochs, 'completed out of', num_of_epochs, 'loss:', epochs_loss)

        correct = tf.equal(tf.arg_max(prediction,1), tf.arg_max(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))


train_model(x)
