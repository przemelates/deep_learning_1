import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt

'''
Constants
'''
IMG_HEIGHT, IMG_WIDTH = 75, 75
EPOCHS = 100
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


labels_combined = labels[:, 0] * 60 + labels[:, 1]


dataset = tf.data.Dataset.from_tensor_slices((data, labels_combined))
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
    
   
    keras.layers.Conv2D(256, (3, 3), padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.2),
    
    
    keras.layers.Flatten(),
    keras.layers.Dense(1024, activation='relu'), 
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(512, activation='relu'),  
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.4),
    
   
    keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)

model.summary()

#Callbacks for better training
callbacks = [
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        'best_clock_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]


history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

'''
Evaluation
'''

score = model.evaluate(test_dataset, verbose=1)
print(f'\nTest Results:')
print(f'Test loss: {score[0]:.4f}')
print(f'Test accuracy: {score[1]:.4f} ({score[1]*100:.2f}%)')


predictions = model.predict(test_dataset)
pred_classes = np.argmax(predictions, axis=1)

#Convert class IDs back to hours and minutes
pred_hours = pred_classes // 60
pred_minutes = pred_classes % 60

#Get true labels
true_classes = np.concatenate([y for x, y in test_dataset], axis=0)
true_hours = true_classes // 60
true_minutes = true_classes % 60

#Calculate separate accuracies
hour_accuracy = np.mean(pred_hours == true_hours)
minute_accuracy = np.mean(pred_minutes == true_minutes)
exact_match = np.mean(pred_classes == true_classes)

print(f'\nDetailed Accuracy:')
print(f'Hour accuracy: {hour_accuracy:.4f} ({hour_accuracy*100:.2f}%)')
print(f'Minute accuracy: {minute_accuracy:.4f} ({minute_accuracy*100:.2f}%)')
print(f'Exact match (both correct): {exact_match:.4f} ({exact_match*100:.2f}%)')

#Calculate "common sense" time difference accuracy
def calculate_time_difference_minutes(true_hours, true_minutes, pred_hours, pred_minutes):
    """
    Calculate absolute time difference in minutes, handling circular clock arithmetic
    """
    #Convert to total minutes since midnight
    true_total_minutes = true_hours * 60 + true_minutes
    pred_total_minutes = pred_hours * 60 + pred_minutes
    
    #Calculate raw difference
    diff = np.abs(true_total_minutes - pred_total_minutes)
    
    #Handle wraparound for 12-hour clock (720 minutes)
    diff = np.minimum(diff, 720 - diff)
    
    return diff
def format_time_difference(minutes):
    """Convert minutes to 'X hours Y minutes' format"""
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    return hours, mins

time_diffs = calculate_time_difference_minutes(true_hours, true_minutes, pred_hours, pred_minutes)

#Calculate average error in hours and minutes format
avg_error_hours, avg_error_mins = format_time_difference(time_diffs.mean())
median_error_hours, median_error_mins = format_time_difference(np.median(time_diffs))

print(f'\n"Common Sense" Time Difference Accuracy:')
print(f'Average error: {avg_error_hours} hour(s) and {avg_error_mins} minute(s)')
print(f'  (Total: {time_diffs.mean():.2f} minutes)')
print(f'Median error: {median_error_hours} hour(s) and {median_error_mins} minute(s)')
print(f'  (Total: {np.median(time_diffs):.2f} minutes)')
print(f'Std deviation: {time_diffs.std():.2f} minutes')

def plot_learning_curves(history, save_path='learning_curves_720.png'):
    """
    Plot training and validation learning curves.
    """

    metrics = [key for key in history.history.keys() if not key.startswith('val_')]
    
    #Determine number of subplots needed
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        #Training metric
        train_values = history.history[metric]
        epochs = range(1, len(train_values) + 1)
        ax.plot(epochs, train_values, 'b-o', label=f'Training {metric}', markersize=4)
        
        #Validation metric 
        val_metric = f'val_{metric}'
        if val_metric in history.history:
            val_values = history.history[val_metric]
            ax.plot(epochs, val_values, 'r-s', label=f'Validation {metric}', markersize=4)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(metric.capitalize(), fontsize=12)
        ax.set_title(f'{metric.capitalize()} vs Epoch', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        #Set y-axis to start from 0 for accuracy metrics
        if 'acc' in metric.lower():
            ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

plot_learning_curves(history)