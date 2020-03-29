from numpy import load
from numpy import expand_dims
from numpy import asarray
from numpy import savez_compressed
from keras.models import load_model

# Extract face embeddings from given face
def get_embedding(model, face_pixels):
	face_pixels = face_pixels.astype('float32')
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std

	samples = expand_dims(face_pixels, axis=0)

	yhat = model.predict(samples)
	return yhat[0]

data = load('distinct-faces-dataset.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print('Loaded: distinct-faces-dataset.npz')

model = load_model('facenet_keras.h5')
print('Loaded: facenet_keras.h5')

# Convert each face in the train set to an embedding
newTrainX = list()

for face_pixels in trainX:
	embedding = get_embedding(model, face_pixels)
	newTrainX.append(embedding)
newTrainX = asarray(newTrainX)

# Convert each face in the test set to an embedding
newTestX = list()

for face_pixels in testX:
	embedding = get_embedding(model, face_pixels)
	newTestX.append(embedding)
newTestX = asarray(newTestX)

savez_compressed('face-embeddings-dataset.npz', newTrainX, trainy, newTestX, testy)
print("\nEmbeddings model: face-embeddings-dataset.npz")
