import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage
from PIL import Image


# main programe
def main():
    # 1
    train_set_x_orig, train_y, test_set_x_orig, test_y = load_dataset()
    
    # 2
    train_X, test_X = pre_processing(train_set_x_orig, test_set_x_orig)
    
    # 3
    model = LogisticRegression(train_X, train_y)
    
    # 4
    model.fit()
    
    # 5
    model.score(test_X, test_y)
    
    
    
def load_dataset():
    # train dataset
    train_dataset = h5py.File("datasets/train_catvnoncat.h5", "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])
    
    # test dataset
    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig

    
def pre_processing(train_set_x_orig, test_set_x_orig):
    print ("Number of training examples: m_train = " + str(train_set_x_orig.shape[0]))
    print ("Number of testing examples: m_test = " + str(test_set_x_orig.shape[0]))
    print ("Height/Width of each image: num_px = " + str(train_set_x_orig.shape[1]))
    
    train_X = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_X = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
    
    # standardize the dataset
    train_X = train_X/255.
    test_X = test_X/255.
    
    return train_X, test_X

    
class LogisticRegression(object):
    def __init__(self, train_X, train_y):
        self.x = train_X
        self.m = self.x.shape[0]
        self.y = train_y
        self.w = np.zeros((self.m, 1))
        self.b = .0
        self.costs = []
        
    def propagate(self):
        # forward
        z = np.dot(self.w.T, self.x) + self.b
        a = 1 / (1 + np.exp(-z))
        cost = -1 / self.m * np.sum(np.dot(self.y, np.log(a).T) + np.dot((1 - self.y), np.log(1 - a).T))
        
        # backward
        dw = 1 / self.m * np.dot(self.x, (a - self.y).T)
        db = 1 / self.m * np.sum((a - self.y))
        
        assert(dw.shape == self.w.shape)
        assert(db.dtype == float)
        cost = np.squeeze(cost)
        assert(cost.shape == ())

        return dw, db, cost
        
    def fit(self, num_iterations=1000, learning_rate=0.1):
        for i in range(num_iterations):
            dw, db, cost = self.propagate()
            self.w = self.w - learning_rate * dw
            self.b = self.b - learning_rate * db

            if i % 100 == 0:
                print(".")
                self.costs.append(cost)
        
    def score(self, test_X, test_y):
        z = np.dot(self.w.T, test_X) + self.b
        test_y_predict = 1 / (1 + np.exp(-z))
        plt.plot(self.costs)
        plt.xlabel("iterations per hundred")
        plt.ylabel("cost")
        plt.title("Learning rate = 2000")
        plt.show()
        print("test accuracy: {} %".format(100 - np.mean(np.abs(test_y_predict - test_y)) * 100))

        
if __name__ == "__main__":
    main()
