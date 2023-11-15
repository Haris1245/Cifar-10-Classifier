
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#* Downloading and formatting data
(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()
train_images, test_images = train_images / 255, test_images/ 255

# * Hard coding class names
class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

#* Testing Dateset
# for i in range(16):
#     plt.subplot(4,4,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i][0]])
    
# plt.show()




# model = keras.Sequential()
# model.add(keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=(32,32,3))) 
# model.add(keras.layers.MaxPooling2D((2,2)))
# model.add(keras.layers.Conv2D(64, (3,3), activation="relu"))
# model.add(keras.layers.MaxPooling2D((2,2)))
# model.add(keras.layers.Conv2D(64, (3,3), activation="relu"))
# model.add(keras.layers.Flatten())
# model.add(keras.layers.Dense(64, activation="relu"))
# model.add(keras.layers.Dense(10, activation="softmax"))

# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.fit(train_images, train_labels, epochs=13, validation_data=(test_images, test_labels), batch_size=32)
# loss,acc = model.evaluate(test_images, test_labels)
# print(f"Accuracy: {round(acc, 4)}, Loss: {round(loss, 4)}" )
# model.save("image_classifier.h5")

model = keras.models.load_model('image_classifier.h5')

img = cv2.imread('dog.jpg')
# No need to convert to grayscale
img = cv2.resize(img, (32, 32))  # Make sure the image is resized to 32x32
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Correct color representation in matplotlib

# Reshape the image for the model prediction
img = img.reshape(1, 32, 32, 3) / 255

prediction = model.predict(img)
index = np.argmax(prediction)
plt.title(f"Prediction is {class_names[index]}")
plt.show()