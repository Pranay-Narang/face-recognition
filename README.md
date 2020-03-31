# Face Recognition
A simple NumPy based face recognition system


## Requirements
* Python 3.x
* Keras (2.3.1).
* Tensorflow (<=2.0).
* MTCNN
* NumPy
* PIL


## Training distinct faces model
Distinct faces refers to all the different people for whom we will be training the model. The number of distinct faces wlll be referred to as the number of classes which should always be > 1.

Below is an example directory structure for storing the images for training the model.
```
faces/
├── train/
│   ├── pranay_narang/
│   ├──├── image-1.jpg
│      ├── image-2.jpg
│      ├── image-3.jpg
│      ├── ...
│   ├── akshat_srivastava/
│   ├──├── image-1.jpg
│      ├── image-2.jpg
│      ├── image-3.jpg
│      ├── ....
│   ├── rudra_dutt/
│   └── tarun_aditya/
└── val/
    ├── pranay_narang/
    ├──├── image-1.jpg
       ├── image-2.jpg
       ├── image-3.jpg
       ├── ...
    ├── akshat_srivastava/
    ├──├── image-1.jpg
       ├── image-2.jpg
       ├── image-3.jpg
       ├── ... 
    ├── rudra_dutt/
    └── tarun_aditya/`
```
Once you create the above directory structure, run
```
$ python3 distinct-face-trainer.py
```
After running for some time (depending on your hardware) it will save the model as `distinct-faces-dataset.npz`


## Training face embeddings model
Face embeddings refer to the features extracted from a specific face which can then be used to compare against other faces and perform facial recognition

Extracting face embeddings requires a pre-trained model. We will be using [facenet_keras.h5](https://drive.google.com/drive/folders/1pwQ3H4aJ8a6yyJHZkTwtjcL4wYWQb7bn) prepared by [Hiroki Taniai](https://github.com/nyoki-mtl)

After downloading the pre-trained model, run
```
$ python3 face-embs-trainer.py
```
After running for some time (again depending on your hardware) it will save the model as `face-embeddings-dataset.npz`


## Running the identifier
The face identifier first loads both `distinct-faces-dataset.npz` and `face-embeddings-dataset.npz` models. After that it takes `needed.jpg` from the local directory, extracts the face from it and stores the face as an array. The array is then used for extracting the embeddings using `facenet_keras.h5` model which are then compared with the existing embeddings and the result is displayed as the predicted class.

Ensure that all three models and *needed.jpg* are present in the directory then run
```
$ python3 face-identifier.py
```
It will display the predicted result.


## Reference
Based on [this](https://machinelearningmastery.com/how-to-develop-a-face-recognition-system-using-facenet-in-keras-and-an-svm-classifier/) article
