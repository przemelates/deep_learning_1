import tensorflow as tf
import keras
import numpy as np

'''
Constants
'''
IMG_HEIGHT, IMG_WIDTH = 75, 75
EPOCHS = 30
BATCH_SIZE = 128

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

#Convert to angles (better for circular data)
#Hours: 0-11 -> 0-330 degrees (30° per hour)
#Minutes: 0-59 -> 0-354 degrees (6° per minute)
hour_angles = (labels[:, 0] % 12) * 30.0  # 0-330
minute_angles = labels[:, 1] * 6.0  # 0-354

#Convert angles to sin/cos (handles circularity)
hour_sin = np.sin(np.radians(hour_angles))
hour_cos = np.cos(np.radians(hour_angles))
minute_sin = np.sin(np.radians(minute_angles))
minute_cos = np.cos(np.radians(minute_angles))

#Stack as 4 outputs: (hour_sin, hour_cos, minute_sin, minute_cos)
labels_circular = np.stack([hour_sin, hour_cos, minute_sin, minute_cos], axis=1)


dataset = tf.data.Dataset.from_tensor_slices((data, labels_circular))
dataset = dataset.shuffle(buffer_size=len(data), seed=42, reshuffle_each_iteration=False)

#Calculate split sizes
total_size = len(data)
train_size = int(0.8 * total_size)
val_size = int(0.1 * total_size)

#Split the dataset
train_dataset = dataset.take(train_size)
remaining = dataset.skip(train_size)
val_dataset = remaining.take(val_size)
test_dataset = remaining.skip(val_size)

#Batch and prefetch for performance
train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

'''
Model architecture 
'''
model = keras.Sequential([
    
    keras.layers.Conv2D(64, (3, 3), padding='same', input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.Conv2D(64, (3, 3), padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.2),
    
   
    keras.layers.Conv2D(128, (3, 3), padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.Conv2D(128, (3, 3), padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.2),
    
 
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.3),
    
    #Output: 4 values (sin/cos for hours and minutes)
    keras.layers.Dense(4, activation='tanh')  #tanh keeps output in [-1, 1]
])

model.compile(
    loss='mse',  
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=['mae']  
)


lr_schedule = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=7,
    restore_best_weights=True,
    verbose=1
)

model.summary()


history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[lr_schedule, early_stop],
    verbose=1
)

'''
Evaluation with proper angle conversion
'''
def convert_predictions_to_time(predictions):
    """Convert sin/cos predictions back to hour and minute"""
    hour_sin, hour_cos, minute_sin, minute_cos = predictions[:, 0], predictions[:, 1], predictions[:, 2], predictions[:, 3]
    
    #Convert back to angles
    hour_angle = np.degrees(np.arctan2(hour_sin, hour_cos)) % 360
    minute_angle = np.degrees(np.arctan2(minute_sin, minute_cos)) % 360
    
    #Convert to hour/minute values
    hours = (hour_angle / 30).astype(int) % 12
    minutes = (minute_angle / 6).astype(int) % 60
    
    return hours, minutes

#Evaluate on test set
predictions = model.predict(test_dataset)
pred_hours, pred_minutes = convert_predictions_to_time(predictions)

#Get true labels from test set
true_labels = np.concatenate([y for x, y in test_dataset], axis=0)
true_hour_angles = np.degrees(np.arctan2(true_labels[:, 0], true_labels[:, 1])) % 360
true_minute_angles = np.degrees(np.arctan2(true_labels[:, 2], true_labels[:, 3])) % 360
true_hours = (true_hour_angles / 30).astype(int) % 12
true_minutes = (true_minute_angles / 6).astype(int) % 60

#Calculate accuracy
hour_accuracy = np.mean(pred_hours == true_hours)
minute_accuracy = np.mean(pred_minutes == true_minutes)
exact_match = np.mean((pred_hours == true_hours) & (pred_minutes == true_minutes))

print(f'\nTest Results:')
print(f'Hour Accuracy: {hour_accuracy:.2%}')
print(f'Minute Accuracy: {minute_accuracy:.2%}')
print(f'Exact Match (both correct): {exact_match:.2%}')

#Calculate average angular error
hour_error = np.abs(pred_hours - true_hours)
hour_error = np.minimum(hour_error, 12 - hour_error)  
minute_error = np.abs(pred_minutes - true_minutes)
minute_error = np.minimum(minute_error, 60 - minute_error)  

print(f'Average Hour Error: {hour_error.mean():.2f} hours')
print(f'Average Minute Error: {minute_error.mean():.2f} minutes')