import cv2
import glob
import os
import re
import multiprocessing
import numpy as np
import pickle
from operator import itemgetter
from keras.applications.vgg16 import VGG16
from keras.models import Model

image_path = 'images/'
extension = '*.jpg'
save_path = 'edges/'
resize_path = 'resize/'
resize = True

chunksize = 64
chunks = 10

def image_generator_index(path, extension='*.jpg', index=None):
    files = glob.glob(os.path.join(path, extension))

    if index is not None:
        files = itemgetter(*index)(files)

    for file in files:
        yield cv2.imread(file)

def image_generator_random(path, extension='*.jpg', n_sample=1000, seed=1):
    np.random.seed(seed)

    files = glob.glob(os.path.join(path, extension))

    files = np.random.choice(files, n_sample)

    for file in files:
        yield cv2.imread(file)

def load_resize(path, resize=(128, 128)):

    img = cv2.imread(path)
    img = cv2.resize(img, resize)

    return img

def edge_invert(img, min_thres=85, max_thres=100, ksize=(5, 5), sigmaX=0):
    """
    detect the edges of an image and invert the colors
    to have white background and black edges

    :param img:
    :return:
    """
    img = cv2.GaussianBlur(img, ksize, sigmaX)
    edges = cv2.Canny(img, threshold1=min_thres, threshold2=max_thres)

    return edges

def load_resize_edge_invert(path, resize=(128, 128)):
    """
    load, process and save images.
    :param path:
    :param save_path:
    :param resize:
    :return:
    """

    img = load_resize(path, resize)

    edges = edge_invert(img)

    return edges

    # cv2.imwrite(save_path, edges)


class CNN_edge_detection(Model):

    def __init__(self, layer='block1_conv2', input_shape=(224, 224, 3)):
        """
        Pass images through VGG16 to detect edges
        """

        self.vgg16 = VGG16(include_top=False, input_shape=input_shape)

        super().__init__(inputs=self.vgg16.input, outputs=self.vgg16.get_layer(layer).output)

def edge_detection():

    image_files = glob.glob(os.path.join(image_path, extension))

    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cores)
    edges = pool.map(load_resize_edge_invert, iterable=image_files, chunksize=chunksize)

    edges = np.array(edges)
    n = len(edges)
    index = n // chunks

    for i in range(chunks):
        with open('edges_{}.pkl'.format(i), 'wb') as f:

            pickle.dump(edges[i*index : (i+1)*index], f)

def batch_resize():
    image_files = glob.glob(os.path.join(image_path, extension))

    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cores)
    img = pool.map(load_resize, image_files, chunksize=chunksize)

    img = np.array(img)
    
    n = len(img)
    index = n // chunks

    for i in range(chunks):
        with open('img_{}.pkl'.format(i), 'wb') as f:
            pickle.dump(img[i * index: (i + 1) * index], f)

if __name__ == '__main__':
    edge_detection()
    # batch_resize()

