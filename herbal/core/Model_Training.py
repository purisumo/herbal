import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
import matplotlib.pyplot as plt
import json

# Load your data
def train_model(label_file, herb_file):
    herbs = np.load(label_file)
    labels = np.load(herb_file)

    # Shuffle and preprocess the data
    s = np.arange(herbs.shape[0])
    np.random.shuffle(s)
    herbs = herbs[s]
    labels = labels[s]

    num_classes = len(np.unique(labels))
    data_length = len(herbs)

    class_mapping = {i: str(label_name) for i, label_name in enumerate(np.unique(labels))}

    # Split the data into training and testing sets
    (x_train, x_test) = herbs[(int)(0.1 * data_length):], herbs[:(int)(0.1 * data_length)]
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    train_length = len(x_train)
    test_length = len(x_test)

    (y_train, y_test) = labels[(int)(0.1 * data_length):], labels[:(int)(0.1 * data_length)]

    # Define the EarlyStopping callback
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        min_delta=0.001,
        patience=1,
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

    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Freeze the weights of the pre-trained layers
    for layer in base_model.layers:
        layer.trainable = False

    # Create a new model on top of the ConvNeXtXLarge base
    model = models.Sequential()
    model.add(base_model)
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    # Train the model
    history = model.fit(x_train, y_train, epochs=100,
                        validation_data=(x_test, y_test), callbacks=[early_stopping, model_checkpoint])

    # Unfreeze some layers for fine-tuning
    for layer in base_model.layers[-10:]:
        layer.trainable = True

    # Recompile the model after unfreezing some layers
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
    model.compile(optimizer=optimizer,
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    fine_tune_epochs = 1  # You need to define this variable

    history_fine_tune = model.fit(x_train, y_train, epochs=fine_tune_epochs,
                                validation_data=(x_test, y_test), callbacks=[early_stopping, model_checkpoint])


    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(test_acc)

    model.save("model_with_transfer_learning.h5")

    save_training_history(history, "training_history.json", num_classes, data_length)
    save_class_history(class_mapping, "class_history.json")

def save_training_history(history, filename, num_classes, data_length):
    history_dict = {
        "training_accuracy": history.history['accuracy'][-1],
        "validation_accuracy": history.history['val_accuracy'][-1],
        "training_loss": history.history['loss'][-1],
        "validation_loss": history.history['val_loss'][-1],
        "num_classes": num_classes,
        "data_length": data_length,
    }

    with open(filename, 'w') as file:
        json.dump(history_dict, file, indent=4)

def save_class_history(class_mapping, filename):
    
    class_mapping_str = {str(key): str(value) for key, value in class_mapping.items()}

    history_dict = {
        "class_mapping": class_mapping_str,
    }

    with open(filename, 'w') as file:
        json.dump(history_dict, file, indent=4)