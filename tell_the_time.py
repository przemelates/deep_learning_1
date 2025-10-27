import tensorflow as tf
import keras
import numpy as np

'''
Constants
'''

IMG_HEIGHT,IMG_WIDTH = 75,75
EPOCHS = 20
BATCH_SIZE = 128
NUM_CLASSES = 720

'''
Data pre-processing
'''

data_path = "/home/przemelates/.vscode/deep_learning_1/images.npy"
labels_path = "/home/przemelates/.vscode/deep_learning_1/labels.npy"

data = np.load(data_path)
labels = np.load(labels_path)
data = data.astype('float32') / 255.0
if len(data.shape) == 3:
    data = np.expand_dims(data, axis=-1)
labels_combined = labels[:, 0] * 61 + labels[:, 1]  


dataset = tf.data.Dataset.from_tensor_slices((data, labels_combined))
dataset = dataset.shuffle(buffer_size=len(data), seed=42, reshuffle_each_iteration=False)

#Calculate split sizes
total_size = len(data)
train_size = int(0.8 * total_size)  
val_size = int(0.1 * total_size)    
test_size = total_size - train_size - val_size  

#Split the dataset
train_dataset = dataset.take(train_size)
remaining = dataset.skip(train_size)
val_dataset = remaining.take(val_size)
test_dataset = remaining.skip(val_size)


#Prefetch for training performance boost
train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

'''
Model architecture and evaluation
'''

model = keras.Sequential()

model.add(keras.layers.Conv2D(32, kernel_size=(3, 3), input_shape=(IMG_HEIGHT,IMG_WIDTH,1), padding='same'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation('silu'))
model.add(keras.layers.Conv2D(32, kernel_size=(3, 3),padding='same'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation('silu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.25))

model.add(keras.layers.Conv2D(64, kernel_size=(3, 3),padding='same'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation('silu'))
model.add(keras.layers.Conv2D(64, kernel_size=(3, 3),padding='same'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation('silu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.25))

model.add(keras.layers.Conv2D(128, kernel_size=(3, 3),padding='same'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation('silu'))
model.add(keras.layers.Conv2D(128, kernel_size=(3, 3),padding='same'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation('silu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.25))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(256, activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Dense(NUM_CLASSES, activation='softmax'))

model.compile(loss=keras.losses.sparse_categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])
model.fit(train_dataset,validation_data=val_dataset,
          epochs=EPOCHS,verbose=1)
score = model.evaluate(test_dataset,verbose=1)

print('Test loss:', score[0])
print('Test accuracy:', score[1])