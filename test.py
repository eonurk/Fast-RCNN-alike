#!/usr/bin/env python3
import numpy as np
from PIL import Image
import scipy.misc as scp
from preprocessing import preprocessing
from train import extractFeatures
from pathlib import Path
import pickle
import os
import selectivesearch as ss
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

modelPath = Path("model.joblib")
model = []
if not modelPath.is_file():
  print("Can't find the model.joblib. Running train.py...")
  os.system("python3 train.py")
with open(str(modelPath), "rb") as fp:   #Pickling
  model = pickle.load(fp)
for i in range(10):
  image = Image.open("test/images/"+str(i)+".JPEG").convert('RGB') # load an image
  image = np.asarray(image) # convert to a numpy array
  img_lbl, regions = ss.selective_search(image)

  candidates = set()
  maxProb = 0
  chosenClass = []
  index = 0
  for r in regions:
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
    # candidates.add(r['rect'])
    imageBox = preprocessing(image[x:x+w,y:y+h])
    boxFeatures = extractFeatures(imageBox)
    currentProb = np.amax(model.predict_proba(boxFeatures))
    if maxProb < currentProb:
      maxProb = currentProb
      chosenRegion = r['rect']
      chosenClass = model.predict(boxFeatures)
    
    index +=1
    print(str(index/50*100))
    if index > 50:
      break
  # draw rectangles on the original image
  fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
  ax.imshow(image)
  x,y,w,h = chosenRegion
  print(x,y,w,h)
  print("Probability: ",maxProb, "\nClass: ",chosenClass )
  rect = mpatches.Rectangle(
    (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
  ax.add_patch(rect)

plt.show()