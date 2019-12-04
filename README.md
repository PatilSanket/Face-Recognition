# Face-Recognition
Face recognition process in this repository is built on Haar cascades from OpenCV. 
The repository consist of facial images of the people you want the model to train on, 
the recogniser code which takes the images as input(I have used PIL to import images from local system, but ImageDataGenerator from Keras can also be used),
the face detector code which detects the faces of the people recogniser trained on and
a yml file which kind of acts as mediator between the codes(It stores the trained features from recognizer code in YML format).

The file structure should look like:
/FaceRecognition
---/TrainingData
   ---/1_1.jpg        (The facial images of people to be trained, starting with unique ID. For Example 1 for Virat Kohli, 2 for Leo Messi etc)
       1_2.jpg
       .
       .
       .
   ---/FaceRecog.py
   ---/FaceDetect.py
   ---/Trainer.yml
   
   
