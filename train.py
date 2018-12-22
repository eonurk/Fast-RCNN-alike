from os import listdir
from preprocessing import preprocessing
from resnet import resnet50
import numpy as np
import torch
import torchvision
import warnings
warnings.filterwarnings("ignore")

import pickle
from pathlib import Path

featuresPath =Path("allFeatures.txt")
labelsPath = Path("labels.txt")

if featuresPath.is_file() and labelsPath.is_file():
  print("Features and Labels are loading...")
  with open(str(featuresPath), "rb") as fp:   # Unpickling
    allFeatures = pickle.load(fp)
  with open(str(labelsPath), "rb") as fp:   # Unpickling
    labels = pickle.load(fp)
else:
  print("Can't find features/labels, calculating...")
  # do the padding, rescaling and normalization
  # at this point make sure that the image is of type np.float32
  path = "train"
  labels = []
  allFeatures = []
  # create model
  model = resnet50()
  idx = 0
  for imageDir in listdir(path):
    for imageName in listdir(path+'/'+imageDir):
      idx+=1
      print(idx)
      pathImg = path + '/' + imageDir + '/' + imageName
      image = preprocessing(pathImg)
      image = np.array(image, dtype="float32")

      # given for pytorch image RGB normalization
      image[:,:,0] -= 0.485 
      image[:,:,0] /= 0.229 
      image[:,:,1] -= 0.456 
      image[:,:,1] /= 0.224 
      image[:,:,2] -= 0.406 
      image[:,:,2] /= 0.225

      image= image.astype(np.float32)
      # we append an augmented dimension to indicate batch_size, which is one
      image = np.reshape(image, [1, 224, 224, 3])

      # model takes as input images of size [batch_size, 3, im_height, im_width]
      image = np.transpose(image, [0, 3, 1, 2])
      # convert the Numpy image to torch.FloatTensor
      image = torch.from_numpy(image)

      # extract features
      feature_vector = model(image)
      # convert the features of type torch.FloatTensor to a Numpy array
      # so that you can either work with them within the sklearn environment
      # or save them as .mat files
      feature_vector = feature_vector.detach().numpy()
      # feature vector : (1,2048)
      allFeatures.append(feature_vector)
      labels.append(imageDir)
      with open(str(featuresPath), "wb") as fp:   #Pickling
        pickle.dump(allFeatures, fp)
      with open(str(labelsPath), "wb") as fp:   #Pickling
        pickle.dump(labels, fp)

allFeatures = np.asarray(allFeatures)
labels = np.asarray(labels)
allFeatures = np.reshape(allFeatures,[allFeatures.shape[0],allFeatures.shape[2]])

from sklearn.svm import SVC
svm = SVC(kernel='linear')
svm.fit(allFeatures,labels)

