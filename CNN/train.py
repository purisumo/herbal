import numpy as np
# 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
# 

herbs=np.load("herbs.npy")
labels=np.load("labels.npy")

s=np.arange(herbs.shape[0])
np.random.shuffle(s)
herbs=herbs[s]
labels=labels[s]

# 
num_classes=len(np.unique(labels))
data_length=len(herbs)

# 
(x_train,x_test)=herbs[(int)(0.1*data_length):],herbs[:(int)(0.1*data_length)]
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255
train_length=len(x_train)
test_length=len(x_test)

# 
(y_train,y_test)=labels[(int)(0.1*data_length):],labels[:(int)(0.1*data_length)]

# Define the EarlyStopping callback
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    min_delta=0.001,  # Reduce the minimum delta for early stopping
    patience=15,      # Increase patience to give more time for improvement
    verbose=1,
    restore_best_weights=True,
)

# Add ModelCheckpoint callback for saving the best model
model_checkpoint = keras.callbacks.ModelCheckpoint(
    filepath="best_model.h5",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1,
)
# CNN https://www.tensorflow.org/tutorials/images/cnn

model = models.Sequential()
model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(50, 50, 3)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.summary()

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=100, 
                    validation_data=(x_test, y_test), callbacks=[early_stopping, model_checkpoint])

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)

print(test_acc)

model.save("model.h5")
