import csv
import tensorflow as tf
import numpy

numpy.random.seed(7)

dataset = numpy.loadtxt("voice.csv", delimiter=",")
test_dataset = numpy.loadtxt("test_voice.csv", delimiter=",")

X = dataset[:,0:20]
Y = dataset[:,20]

test_X = test_dataset[:,0:20]
test_Y = test_dataset[:,20]

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(20, activation='relu'))
#model.add(tf.keras.layers.Dense(10, activation='relu'))
model.add(tf.keras.layers.Dense(18, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])

model.fit(X, Y, epochs=50)

scores = model.evaluate(test_X, test_Y)
print("\nDla danych testowych - %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# print(dataset)
# print(X)
# print(Y)
