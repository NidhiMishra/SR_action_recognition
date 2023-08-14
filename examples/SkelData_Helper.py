import numpy as np
import pickle
import os

n_classes = 21

def one_hot_encode(name):
    vec = [x in name for x in (
    'A001', 'A006', 'A010', 'A011', 'A012', 'A014', 'A015', 'A018', 'A019', 'A023', 'A025', 'A028', 'A029', 'A031',
    'A032', 'A033', 'A035', 'A036', 'A038', 'A039', 'A041')]
    hot_one = np.asarray(vec)
    hot_one.astype(int)
    return hot_one


class SkelData_Helper:

    _train_names = []
    _test_names = []
    _data_mean = np.zeros(86016, dtype=np.float64)
    _train_x = []
    _train_y = []

    def __init__(self):
        self.load_train_names()
        self.load_test_names()
        self.load_data_mean()

    def load_train_names(self):
        with open("train_names.pkl", "rb") as fp:  # Unpickling
            self._train_names = pickle.load(fp)

        print("Training names loaded",len(self._train_names))
        return self._train_names

    def load_test_names(self):
        with open("test_names.pkl", "rb") as fp:  # Unpickling
            self._test_names = pickle.load(fp)

        print("Testing names loaded",len(self._test_names))
        return self._test_names

    def load_test_data(self):
        with open("test_names.pkl", "rb") as fp:  # Unpickling
            test_names = pickle.load(fp)

        test_x = np.zeros((len(test_names), 86016), dtype=np.float64)
        test_y = np.zeros((len(test_names), n_classes), dtype=np.int)

        iter = 0
        for file in test_names:
            name = os.path.basename(file)
            s = np.load(os.path.join(os.path.abspath('.\\Nadine_Features_05122017'), name))
            #s = np.load(file)
            test_x[iter, :] = np.reshape(s, 86016) #- self._data_mean
            test_y[iter, :] = one_hot_encode(file)
            iter += 1

        print("Test data loaded")
        return test_x, test_y

    def load_train_data(self):
        with open("train_names.pkl", "rb") as fp:  # Unpickling
            train_names = pickle.load(fp)

        self._train_x = np.zeros((len(train_names), 86016), dtype=np.float64)
        self._train_y = np.zeros((len(train_names), n_classes), dtype=np.int)

        iter = 0
        for file in train_names:
            s = np.load(file)
            self._train_x[iter, :] = np.reshape(s, 86016) #- self._data_mean
            self._train_y[iter, :] = one_hot_encode(file)
            iter += 1

        print("Train data loaded")
        return self._train_x, self._train_y

    def load_data_mean(self):
        with open("data_mean.pkl", "rb") as fp:  # Unpickling
            self._data_mean = pickle.load(fp)

        print("Data mean loaded")
        return self._data_mean

    def RAM_next_batch(self,start,batch_size):

        total_length = len(self._train_names)
        if (start+batch_size >= total_length):
            allocate_for = total_length-start
        else:
            allocate_for = batch_size

        train_x = self._train_x[start:start + allocate_for, :]
        train_y = self._train_y[start:start + allocate_for, :]

        return train_x, train_y

    def next_batch(self,start,batch_size):
        total_length = len(self._train_names)
        if (start+batch_size >= total_length):
            allocate_for = total_length-start
        else:
            allocate_for = batch_size
        train_x = np.zeros((allocate_for,86016),dtype=np.float64)
        train_y = np.zeros((allocate_for, n_classes), dtype=np.float64)

        iter = 0
        for name in self._train_names[start:start+allocate_for]:
            t = np.load(name)
            train_x[iter,:] = np.reshape(t,86016) - self._data_mean
            train_y[iter,:] = one_hot_encode(name)
            iter += 1

        return train_x, train_y