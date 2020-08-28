# bad-good-road-classifier

According to the requirements, there are 5 CNN architectures to be implemented in order to get the best results for the binary classification problem. 

ResNet50
Inception V3
Xception
VGG19
VGG16

All the models except VGG16 gave over accuracy of 96% for both training and validation. VGG16 gave the worst accuracy. Therefore, I proposed a new architecture which is InceptionResNetV2 instead of VGG16.

Instructions to run


Download the source code attached with the delivery. Extract it and install the libraries mentioned in the requirements.txt file. 

Download all the weight files using this url and place them in the weights folder.

Now, you need to give the path where the dataset is located in your computer in line 15 of main.py file. 

Then, you are ready to go. You can evaluate the model by running,

						python main.py

If you want to train the models, uncomment from line 45 to line 50 in main.py.

