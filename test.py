#!/usr/bin/env python3
from sklearn.metrics import confusion_matrix
from preprocessing import preprocessing
from collections import namedtuple
from train import extractFeatures
from pathlib import Path
from PIL import Image
import os
import pdb
import sys
import pickle
import progressbar
import numpy as np
import pandas as pd
import scipy.misc as scp
import selectivesearch as ss
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# https://stackoverflow.com/questions/27152904/calculate-overlapped-area-between-two-rectangles
def area(a, b):  # returns None if rectangles don't intersect
    dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
    if (dx>=0) and (dy>=0):
        return dx*dy


modelPath = Path("model.joblib")
model = []
if not modelPath.is_file():
  print("Can't find the model.joblib. Running train.py...")
  os.system("python3 train.py")
with open(str(modelPath), "rb") as fp:   #Pickling
  model = pickle.load(fp)

df=pd.read_csv('test/bounding_box.txt', sep=',',header=None)
trueLabels = df[0].values
trueBoxes = df.drop(0,axis=1).values
predictedLabels = []
predictedBoxes = []
localization = []
Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

pbar = progressbar.ProgressBar()
testImgCount = 100
for i in range(testImgCount):
  image = Image.open("test/images/"+str(i)+".JPEG").convert('RGB') # load an image
  image = np.asarray(image) # convert to a numpy array
  _, regions = ss.selective_search(image)

  candidates = set()
  maxProb = 0
  chosenClass = []
  print("\nImage " + str(i) +".JPEG is processing...\n")
  pbar.start()
  for idx, r in enumerate(regions):
    pbar.update(idx/len(regions)*100)
    # excluding same rectangle (with different segments)    
    if r['rect'] in candidates:
      continue
    # excluding regions smaller than 2000 pixels
    if r['size'] < 2500:
      continue
    # distorted rects
    x, y, w, h = r['rect']
    if w / h > 1.2 or h / w > 1.2:
      continue
    candidates.add(r['rect'])
    imageBox = preprocessing(image[x:x+w,y:y+h])
    boxFeatures = extractFeatures(imageBox)
    currentProb = np.amax(model.predict_proba(boxFeatures))
    if maxProb < currentProb:
      maxProb = currentProb
      cr = r['rect'] # chosen region
      chosenClass = model.predict(boxFeatures)
  predictedLabels.append(chosenClass)
  predictedBoxes.append(cr)

  pbar.finish()

  if (chosenClass == trueLabels[i]):
    r_pred = Rectangle(cr[0], cr[1], cr[0]+cr[2], cr[1]+cr[3])
    r_true = Rectangle(trueBoxes[i][0], trueBoxes[i][1], trueBoxes[i][0]+trueBoxes[i][2], trueBoxes[i][1]+trueBoxes[i][3])
    intersection = area(r_true,r_pred)
    union = (cr[2]*cr[3] + trueBoxes[i][2]*trueBoxes[i][3]) - intersection
    localization.append(intersection/union)
  else :
    localization.append(-1) # not a correctly classified image
  # pdb.set_trace() # dont forget to comment to take run!

print(localization)
print(confusion_matrix(trueLabels[0:testImgCount], predictedLabels))

#### TO SEE THE BOUNDING BOXES UNCOMMENT THIS PART ####
#   # draw rectangles on the original image
#   fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
#   ax.imshow(image)
#   x,y,w,h = chosenRegion
#   print(x,y,w,h)
#   print("Probability: ",maxProb, "\nClass: ",chosenClass )
#   rect = mpatches.Rectangle(
#     (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
#   ax.add_patch(rect)

# plt.show()