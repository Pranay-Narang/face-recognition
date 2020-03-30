from random import choice
from numpy import load
from numpy import expand_dims
from numpy import asarray
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from matplotlib import pyplot
from PIL import Image
from mtcnn.mtcnn import MTCNN
from keras.models import load_model

# Insert the distinct faces model
data = load('distinct-faces-dataset.npz')
testX_faces = data['arr_2']

# Insert face embeddings model
data = load('face-embeddings-dataset.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']

in_encoder = Normalizer(norm='l2')

trainX = in_encoder.transform(trainX)
testX = in_encoder.transform(testX)

out_encoder = LabelEncoder()
out_encoder.fit(trainy)

trainy = out_encoder.transform(trainy)
testy = out_encoder.transform(testy)

# Configure the model
model = SVC(kernel='linear', probability=True)
model.fit(trainX, trainy)

def extract_face(filename, required_size=(160, 160)):
	image = Image.open(filename)
	image = image.convert('RGB')
	pixels = asarray(image)

	detector = MTCNN()
	results = detector.detect_faces(pixels)

	x1, y1, width, height = results[0]['box']
	x1, y1 = abs(x1), abs(y1)
	x2, y2 = x1 + width, y1 + height

	face = pixels[y1:y2, x1:x2]

	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = asarray(image)

	return face_array

face_array = extract_face('needed.jpg')

def get_embedding(model, face_pixels):
	face_pixels = face_pixels.astype('float32')
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std

	samples = expand_dims(face_pixels, axis=0)

	yhat = model.predict(samples)
	return yhat[0]

farray_embs = get_embedding(load_model('facenet_keras.h5'), face_array)

# Predict the face from random image
samples = expand_dims(farray_embs, axis=0)
yhat_class = model.predict(samples)
yhat_prob = model.predict_proba(samples)

# Print out the predicted name
class_index = yhat_class[0]
class_probability = yhat_prob[0,class_index] * 100
predict_names = out_encoder.inverse_transform(yhat_class)
print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
