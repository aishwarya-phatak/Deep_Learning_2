import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import mnist

#load the dataset
data = mnist.load_data()
print(type(data))

(X_train,y_train),(X_test,y_test) = data            #load data method returns two tuples 1 for training anf 1 for testing

model = Sequential()

model.add(Flatten(input_shape=(28, 28)))            #convert into 1-D

model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10)

test_loss, test_acc = model.evaluate(X_test, y_test)

print('Test accuracy:', test_acc)
print('Test loss:', test_loss)

print("---------model summary------------")
print(model.summary())

#image observation- for study purpose
first_image = X_train[0]
first_label = y_train[0]

plt.imshow(first_image,cmap='gray')
plt.title(f"label : {first_label}")
plt.axis('off')
plt.show()

predictions = model.predict(X_test)
print(predictions)

print("predictions : ",np.argmax(predictions[0]))
print("actual values : ",(y_test[0]))