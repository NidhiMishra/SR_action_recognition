#'drink water', 'pickup', 'clapping', 'reading', 'writing', 'wear jacket', 'take off jacket', 'wear on glasses'
#'take off glasses', 'hand waving', 'put something inside pocket / take out something from pocket', 'make a phone call/answer phone',
#'playing with phone/tablet', 'pointing to something with finger', 'taking a selfie', 'check time (from watch)',
#'nod head/bow', 'shake head', 'salute', 'put the palms together', 'sneeze/cough'
import numpy as np
import pickle
import os
import SkelData_Helper
import tensorflow as tf
import glob
import SkelNet_train
import re
import time
import random
import test_lib

def pred_to_action(index):
    Action_Name = ['drink water', 'pickup', 'clapping', 'reading', 'writing', 'wear jacket', 'take off jacket', 'wear on glasses', 'take off glasses', 'hand waving', 'put something inside pocket / take out something from pocket', 'make a phone call/answer phone', 'playing with phone/tablet', 'pointing to something with finger', 'taking a selfie', 'check time (from watch)', 'nod head/bow', 'shake head', 'salute', 'put the palms together', 'sneeze/cough']
    return Action_Name[index]


#For random class labels
labels = ['A001', 'A006', 'A010', 'A011', 'A012', 'A014', 'A015', 'A018', 'A019', 'A023', 'A025', 'A028', 'A029', 'A031',
    'A032', 'A033', 'A035', 'A036', 'A038', 'A039', 'A041']
# input data handling from SkelData_Helper class
data = SkelData_Helper.SkelData_Helper()
test_x, test_y = data.load_test_data()

prediction = SkelNet_train.Skeleton_CNN(SkelNet_train.x)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=SkelNet_train.y))
# By Default, learning rate set to 0.001
optimizer = tf.train.AdamOptimizer().minimize(cost)

saver = tf.train.Saver()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#For running the features from kinect in Feature_Share_dir
Feature_Share_dir = os.path.abspath("./Feature_Share")
with tf.Session(config=config) as sess:
    # Import saved model from location 'loc' into local graph

    checkpoint_file = os.path.abspath("./train_model/Skel_Net_35.ckpt")
    #checkpoint_file = os.path.abspath(".\\train_model\\Skel_Net_35.ckpt")
    saver.restore(sess, checkpoint_file)

    Act_end_bool = not os.path.exists(os.path.join(Feature_Share_dir, 'Action_end.txt'))
    while Act_end_bool:
        print("Inside loop 1")
        Act_pred_bool = not os.path.exists(os.path.join(Feature_Share_dir, 'Action_prediction.txt'))
        while Act_pred_bool and Act_end_bool:
            #print("Inside loop 2")
            Act_pred_bool = not os.path.exists(os.path.join(Feature_Share_dir, 'Action_prediction.txt'))
            Act_end_bool = not os.path.exists(os.path.join(Feature_Share_dir, 'Action_end.txt'))

        Act_pred_bool = not os.path.exists(os.path.join(Feature_Share_dir, 'Action_prediction.txt'))
        if not Act_pred_bool:
            check_x = test_x[0:1000, :]
            check_y = test_y[0:1000, :]
            feature_files = glob.glob(os.path.join(Feature_Share_dir, '*.npy'))
            Skel_id = []
            for feature in feature_files:
                head, tail = os.path.split(feature)
                #identify Skel id from .npy file name
                id = re.findall(r'\d+', tail)
                iter = int(id[0])
                Skel_id.append(iter)
                print(Skel_id)

                s = np.load(feature)
                check_x[iter, :] = np.reshape(s, 86016)
                #check_y[iter, :] = SkelData_Helper.one_hot_encode('S001C001P001R001A014_0Skel')
                check_y[iter, :] = SkelData_Helper.one_hot_encode(random.choice(labels))

            #_, c, p = sess.run([optimizer, cost, prediction], feed_dict={SkelNet_train.x: check_x, SkelNet_train.y: check_y})
            c, p = sess.run([cost, prediction], feed_dict={SkelNet_train.x: check_x, SkelNet_train.y: check_y})

            # pred_file = open("C:\\xampp\\htdocs\\Reader\\Action_predictions.txt", "w")

            pred_file = open(os.path.join(test_lib.xampp_path, "Action_predictions.txt"), "w")
            for iter in Skel_id:
                print(str(iter), ") ", np.argmax(p[iter]))
                print(pred_to_action(np.argmax(p[iter])))
                pred_file.write(pred_to_action(np.argmax(p[iter])))

            pred_file.close()
            os.remove(os.path.join(Feature_Share_dir, 'Action_prediction.txt'))
            for feature in feature_files:
                os.remove(feature)

        Act_end_bool = not os.path.exists(os.path.join(Feature_Share_dir, 'Action_end.txt'))
    time.sleep(3)
    os.remove(os.path.join(Feature_Share_dir, 'Action_end.txt'))
