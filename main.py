import tensorflow
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.layers import GlobalMaxPooling2D
import cv2
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
model=ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
model.trainble=False
model=tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])
model.summary()

img=cv2.imread('1636.jpg')
img=cv2.resize(img, (224,224))
img=np.array(img)

img.shape

expand_img=np.expand_dims(img, axis=0)
expand_img.shape

pre_img=preprocess_input(expand_img)

result=model.predict(pre_img).flatten()
normalized=result/norm(result)
normalized.shape


def extract_feature(img_path, model):
    img=cv2.imread(img_path)
    img=cv2.resize(img, (224,224))
    img=np.array(img)
    expand_img=np.expand_dims(img, axis=0)
    pre_img=preprocess_input(expand_img)
    result=model.predict(pre_img).flatten()
    normalized=result/norm(result)
    return normalized
extract_feature('1636.jpg', model)


filename=[]
feature_list=[]
for file in os.listdir("Dataset"):
    filename.append(os.path.join('Dataset', file))


for file in tqdm(filename) :
    feature_list.append(extract_feature(file, model))


import pickle
pickle.dump(feature_list, open('featurevactor.pkl', 'wb'))
pickle.dump(filename, open('filename.pkl','wb'))




