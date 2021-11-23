import glob
import os
import pickle
import numpy as np
from pathlib import Path

import torchvision


def main():
    print('Making datasets directory if does not exist already\n')
    Path("/.datasets").mkdir(parents=True, exist_ok=True)

    print('Downloading cifar dataset in raw form from torchvision.datasets \n')
    trainset = torchvision.datasets.CIFAR10(root='./datasets', train=True, download=True)

    data_dir = './datasets/cifar-10-batches-py'

    train_files = glob.glob(os.path.join(data_dir, 'data_batch*'))
    test_files = glob.glob(os.path.join(data_dir, 'test_batch*'))

    # all_files = all_files + test_files
    assert train_files, 'File paths to load are empty. Please ensure downloaded files are in {}'.format(data_dir)
    assert test_files, 'File paths to load are empty. Please ensure downloaded files are in {}'.format(data_dir)


    def unpickle(file):
        with open(file, 'rb') as fo:
            dic = pickle.load(fo, encoding='bytes')
        return dic


    # label
    # 0:airplane, 1:automobile, 2:bird. 3:cat, 4:deer, 5:dog, 6:frog, 7:horse, 8:ship, 9:truck
    X_train = []
    y_train = []

    X_test = []
    y_test = []

    print('Load and process files - Train\n')


    for file in train_files:
        print('Processing train file - ', file)
        ret = unpickle(file)

        for i, arr in enumerate(ret[b'data']):
            img = np.reshape(arr, (3, 32, 32))
            img = img.transpose(1, 2, 0)
            X_train.append(img)
            y_train.append(ret[b'labels'][i])

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    print('Train files shape')
    print(X_train.shape)

    print('Train labeles shape')
    print(y_train.shape)



    print('Load and process files - Test\n')



    for file in test_files:
        print('Processing test file - ', file)
        ret = unpickle(file)

        for i, arr in enumerate(ret[b'data']):
            img = np.reshape(arr, (3, 32, 32))
            img = img.transpose(1, 2, 0)
            X_test.append(img)
            y_test.append(ret[b'labels'][i])

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    print('Test files shape')
    print(X_test.shape)

    print('Test labeles shape')
    print(y_test.shape)

    print('Saving .npz file with X_train, y_train, X_test, y_test keys\n')

    np.savez(os.path.join('./datasets', 'cifar.npz'), X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

    print('Saved at .datasets/cifar.npz')


if __name__ == "__main__":
    main()
