import numpy as np
import pickle
import os
import SkelData_Helper
import tensorflow as tf
import glob

#Feature input size
n_inputsize = 86016

#Number of classes
n_classes = 21

#parameters of the network
batch_size = 100

# Fully-connected layer.
fc_size = 512             # Number of neurons in fully-connected layer.

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_inputsize])
y = tf.placeholder(tf.float32, [None, n_classes])


def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def new_biases(length):
    #equivalent to y intercept
    #constant value carried over across matrix math
    return tf.Variable(tf.constant(0.05, shape=[length]))


def new_fc_layer(input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 use_relu=True): # Use Rectified Linear Unit (ReLU)?

    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(input, weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer


def Skeleton_CNN(x):

    layer_fc1 = new_fc_layer(input=x,
                             num_inputs=n_inputsize,
                             num_outputs=fc_size,
                             use_relu=False)

    layer_bn1 = tf.contrib.layers.batch_norm(layer_fc1,
                                      center=True, scale=True,
                                      is_training=True,
                                      scope='bn')
    layer_bn1 = tf.nn.relu(layer_bn1)

    layer_fc2 = new_fc_layer(input=layer_bn1,
                             num_inputs=fc_size,
                             num_outputs=n_classes,
                             use_relu=False)

    return layer_fc2


def train_neural_network(x):
    print("Training Neural Network")
    # input data handling from SkelData_Helper class
    data = SkelData_Helper.SkelData_Helper()
    test_x, test_y = data.load_test_data()
    data.load_train_data()

    prediction = Skeleton_CNN(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    #By Default, learning rate set to 0.001
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    n_epochs = 1
    #n_epochs = 35
    Max_Accuracy = 0
    Epoch_max = 0
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(n_epochs):
            epoch_loss = 0
            for batch_iter in range(int(len(data._train_names) / batch_size) + 1):
                epoch_x, epoch_y = data.RAM_next_batch((batch_iter * batch_size), batch_size)
                #epoch_x, epoch_y = data.next_batch((batch_iter * batch_size),batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch + 1, 'completed out of', n_epochs, 'loss:', epoch_loss)
            #print((epoch+1) % 5)
            #if ((epoch+1) % 5) == 0:
                #print("Saving model")
            #    checkpoint_file = os.path.abspath("./train_model/Skel_Net_" + str(epoch+1) + ".ckpt")
            #    print("Saving model ", checkpoint_file)
            #    saver.save(sess, checkpoint_file)

            test_batch_size = 1000
            Final_accuracy = 0
            for batch_iter in range(int(len(data._test_names) / test_batch_size) + 1):
                correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

                start = (batch_iter * test_batch_size)
                if (start + test_batch_size >= len(data._test_names)):
                    allocate_for = len(data._test_names) - start
                else:
                    allocate_for = test_batch_size

                Ac = accuracy.eval({x: test_x[start:start+allocate_for, :], y: test_y[start:start+allocate_for, :]})
                Final_accuracy += int(Ac * allocate_for)
                print('Accuracy:', Ac)

            Final_accuracy = Final_accuracy/len(data._test_names)
            print('Final Accuracy:', Final_accuracy)
            
            if Max_Accuracy < Final_accuracy:
                Max_Accuracy = Final_accuracy
                Epoch_max = epoch + 1

    print("Max_Accuracy:", Max_Accuracy, "at", Epoch_max)


#train_neural_network(x)
#sample_test()

def test_neural_network(x):
    print("Testing")
    # input data handling from SkelData_Helper class
    data = SkelData_Helper.SkelData_Helper()
    test_x, test_y = data.load_test_data()

    prediction = Skeleton_CNN(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    # By Default, learning rate set to 0.001
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        checkpoint_file = os.path.abspath("./train_model/Skel_Net_35.ckpt")
        saver.restore(sess, checkpoint_file)

        test_batch_size = 100
        Final_accuracy = 0
        correct_prediction = np.zeros(len(data._test_names), dtype= np.int)

        for batch_iter in range(int(len(data._test_names) / test_batch_size) + 1):
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

            start = (batch_iter * test_batch_size)
            if (start + test_batch_size >= len(data._test_names)):
                allocate_for = len(data._test_names) - start
            else:
                allocate_for = test_batch_size

            Ac = accuracy.eval({x: test_x[start:start + allocate_for, :], y: test_y[start:start + allocate_for, :]})
            print(int(Ac * allocate_for))
            Final_accuracy += int(Ac * allocate_for)
            print('Accuracy:', Ac)
            f  = test_x[start:start + allocate_for, :]

            _, c, p = sess.run([optimizer, cost, prediction], feed_dict={x: test_x[start:start + allocate_for, :], y: test_y[start:start + allocate_for, :]})
            r = test_y[start:start + allocate_for, :]
            for iter in range(allocate_for):
                print(str(iter), ") ", np.argmax(p[iter]), np.argmax(r[iter]))
                if np.argmax(p[iter]) == np.argmax(r[iter]):
                    correct_prediction[start + iter] = 1

            print(p.shape)
        Final_accuracy = Final_accuracy / len(data._test_names)
        print('Final Accuracy:', Final_accuracy)

#test_neural_network(x)