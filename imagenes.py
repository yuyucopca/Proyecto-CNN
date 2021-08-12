from glob import glob
from scipy.io import loadmat
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.optimizers import SGD

#Files
files = np.array(sorted(glob("jpg/*")))

#Targets
mat = loadmat('vgg102flowers_targets.mat')
labels = mat['labels'][0] - 1

#Random permute(we can also shuffle the dataset)
idx = np.random.permutation(len(files))
files = files[idx]
labels = labels[idx]

#Load classes names
#From: https//github.com/jimgoo/caffe-oxford102
names = ['pink primrose', 'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea', 'english marigold', 'tiger lily', 
'moon orchid', 'bird of paradise', 'monkshood', 'globe thistle', 'snapdragon', 'king protea', 'spear thistle', 'yellow iris', 
'globe-flower', 'purple coneflower', 'peruvian lily', 'balloon flower', 'giant white arum lily', 'fire lily', 'pincushion flower', 
'fritillary', 'red ginger', 'grape hyacinth', 'corn poppy', 'prince of wales feathers', 'stemless gentian', 'artichoke', 'sweet william', 
'carnation', 'garden phlox', 'love in the mist', 'mexican aster', 'alpine sea holly', 'ruby-lipped cattleya', 'cape flower', 'great masterwort', 
'siam tulip', 'lenten rose', 'barbeton daisy', 'daffodil', 'sword lily', 'poinsettia', 'bolero deep blue', 'wallflower', 'marigold', 
'buttercup', 'oxeye daisy', 'common dandelion', 'petunia', 'wild pansy', 'primula', 'sunflower', 'pelargonium', 'bishop of llandaff', 
'gaura', 'geranium', 'orange dahlia', 'pink-yellow dahlia?', 'cautleya spicata', 'japanese anemone', 'black-eyed susan', 'silverbush', 
'californian poppy', 'osteospermum', 'spring crocus', 'bearded iris', 'windflower', 'tree poppy', 'gazania', 'azalea', 'water lily', 
'rose', 'thorn apple', 'morning glory', 'passion flower', 'lotus', 'toad lily', 'anthurium', 'frangipani', 'clematis', 'hibiscus', 
'columbine', 'desert-rose', 'tree mallow', 'magnolia', 'cyclamen ', 'watercress', 'canna lily', 'hippeastrum ', 'bee balm', 'ball moss', 
'foxglove', 'bougainvillea', 'camellia', 'mallow', 'mexican petunia', 'bromelia', 'blanket flower', 'trumpet creeper', 'blackberry lily']

print("Loaded %d files." % len(files))

#Train/valid/test split
train_valid_files, test_files, train_valid_labels, test_labels = train_test_split(files, labels, test_size=0.2, random_state=1234, stratify=labels)
train_files, valid_files, train_labels, valid_labels = train_test_split(train_valid_files, train_valid_labels, test_size=0.25, random_state=5678, stratify=train_valid_labels)

print("Train: ", train_files.shape)
print("Valid: ", valid_files.shape)
print("Test: ", test_files.shape)

del train_valid_files, train_valid_labels

train_frame = pd.DataFrame(np.array([train_files, train_labels]).T, columns=['files', 'labels'])
valid_frame = pd.DataFrame(np.array([valid_files, valid_labels]).T, columns=['files', 'labels'])
test_frame = pd.DataFrame(np.array([test_files, test_labels]).T, columns=['files', 'labels'])

#print(train_frame)

#Cuando se define ImageDateGenerator se incluye el preprocesamiento de las imagenes
train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

#Indicación de como y donde leer la info
train_iter = train_datagen.flow_from_dataframe(train_frame, x_col='files', y_col='labels', target_size=(100, 120), class_mode='categorical', batch_size=32, shuffle=True)
valid_iter = valid_datagen.flow_from_dataframe(valid_frame, x_col='files', y_col='labels', target_size=(100, 120), class_mode='categorical', batch_size=32, shuffle=False)
test_iter = test_datagen.flow_from_dataframe(test_frame, x_col='files', y_col='labels', target_size=(100, 120), class_mode='categorical', batch_size=32, shuffle=False)

#print(test_iter.image_shape)#(target_size, 3 canales RGB) (100, 120, 3) se modificó el tamaño de todas las imágenes a 100x120x3

#Comienza CNN (Alex-NET)
#Los puntos que incluyen la red son:
#-Bloques convolucionales 
#   -Capas convolucionales (funcion de activacion ReLU)
#   -Capa Max Pooling
#-Capas densas o Fully conected (funcion de activacion ReLU)
#-Capa de salida con softmax

#Se define el modelo lineal
model = Sequential()

#Primera capa. Se especifica el tamaño de la entrada
model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(100,120,3)))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

#Para pasar de la capa convolucional a la densa se nesecita "aplanar" la salida (flatten)
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))

#Se crea la capa de salida 
model.add(Dense(102, activation='softmax'))