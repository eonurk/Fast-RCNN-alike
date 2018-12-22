from preprocessing import preprocessing
from resnet import resnet50
from os import listdir
from PIL import Image
import numpy as np
import torchvision
import warnings
import torch
warnings.filterwarnings("ignore")

import pickle
from pathlib import Path

if __name__ == "__main__":

  featuresPath = Path("allFeatures.txt")
  labelsPath = Path("labels.txt")
  modelPath = Path("model.joblib")

  if featuresPath.is_file() and labelsPath.is_file():
    print("Features and labels are loading...")
    with open(str(featuresPath), "rb") as fp:   # Unpickling
      allFeatures = pickle.load(fp)
    with open(str(labelsPath), "rb") as fp:   # Unpickling
      labels = pickle.load(fp)
  else:
    print("Can't find model/features/labels, calculating...")
    # do the padding, rescaling and normalization
    # at this point make sure that the image is of type np.float32
    path = "train"
    labels = []
    allFeatures = []
    idx = 0
    for imageDir in listdir(path):
      for imageName in listdir(path+'/'+imageDir):
        idx+=1
        print(idx)
        pathImg = path + '/' + imageDir + '/' + imageName
        image = Image.open(pathImg).convert('RGB') # load an image
        image = np.asarray(image) # convert to a numpy array
        image = preprocessing(image)
        features = extractFeatures(image)
        allFeatures.append(features)
        labels.append(imageDir)
    with open(str(featuresPath), "wb") as fp:   #Pickling
      pickle.dump(allFeatures, fp)
    with open(str(labelsPath), "wb") as fp:   #Pickling
      pickle.dump(labels, fp)

  allFeatures = np.asarray(allFeatures)
  labels = np.asarray(labels)
  allFeatures = np.reshape(allFeatures,[allFeatures.shape[0],allFeatures.shape[2]])

  from sklearn.svm import SVC
  svm = SVC(kernel='linear',probability=True)
  svm.fit(allFeatures,labels)

  print("Model is saving...")
  with open(str(modelPath), "wb") as fp:   #Pickling
    pickle.dump(svm, fp)
  print("Done. (train.py)")

def extractFeatures(image):
  model = resnet50()
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
  image = torch.from_numpy(image)
  feature_vector = model(image)
  feature_vector = feature_vector.detach().numpy()
  return feature_vector
