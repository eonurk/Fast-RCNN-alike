# CS484-Project
Fast RCNN alike approach to ImageNet subset

## Python Version

###Prerequisites:
	pip install -r requirements.txt
  
### To obtain the results in the report:

Create a folder called "test", put the bounding_box.txt into it and the images into "test/images" folder with name X.JPEG (0.JPEG, 1.JPEG etc.)
Then run,
```
python3 test.py
```
This will give you the localization and confusion matrix results. 

To obtain images given in the report run (you need to create test folder first):
```
python3 visualize.py
```
