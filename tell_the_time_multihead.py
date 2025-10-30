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
NUM_HOUR_CLASSES = 12  
NUM_MINUTE_CLASSES = 60  

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

#Keep labels separate for multi-output
hour_labels = labels[:, 0]  
minute_labels = labels[:, 1]  

print(f"Hour labels range: {hour_labels.min()} to {hour_labels.max()}")
print(f"Minute labels range: {minute_labels.min()} to {minute_labels.max()}")
print(f"Unique hours: {len(np.unique(hour_labels))}")
print(f"Unique minutes: {len(np.unique(minute_labels))}")

#Create dataset with tuple of labels
dataset = tf.data.Dataset.from_tensor_slices((data, (hour_labels, minute_labels)))
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

#Batch and prefetch
train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

print(f"\nDataset splits:")
print(f"Training samples: {train_size}")
print(f"Validation samples: {val_size}")
print(f"Test samples: {total_size - train_size - val_size}")

'''
Multi-head model architecture
'''
#Shared convolutional base
inputs = keras.layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1))


x = keras.layers.Conv2D(64, (3, 3), padding='same')(inputs)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Activation('relu')(x)
x = keras.layers.Conv2D(64, (3, 3), padding='same')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Activation('relu')(x)
x = keras.layers.MaxPooling2D((2, 2))(x)
x = keras.layers.Dropout(0.2)(x)


x = keras.layers.Conv2D(128, (3, 3), padding='same')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Activation('relu')(x)
x = keras.layers.Conv2D(128, (3, 3), padding='same')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Activation('relu')(x)
x = keras.layers.MaxPooling2D((2, 2))(x)
x = keras.layers.Dropout(0.2)(x)


x = keras.layers.Conv2D(256, (3, 3), padding='same')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Activation('relu')(x)
x = keras.layers.MaxPooling2D((2, 2))(x)
x = keras.layers.Dropout(0.2)(x)

#Shared dense layers
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(512, activation='relu')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Dropout(0.4)(x)

#Split into two branches
# Hour head
hour_branch = keras.layers.Dense(128, activation='relu', name='hour_dense')(x)
hour_branch = keras.layers.Dropout(0.3, name='hour_dropout')(hour_branch)
hour_output = keras.layers.Dense(NUM_HOUR_CLASSES, activation='softmax', name='hour_output')(hour_branch)

# Minute head
minute_branch = keras.layers.Dense(256, activation='relu', name='minute_dense')(x)
minute_branch = keras.layers.Dropout(0.3, name='minute_dropout')(minute_branch)
minute_output = keras.layers.Dense(NUM_MINUTE_CLASSES, activation='softmax', name='minute_output')(minute_branch)

#Create model with two outputs
model = keras.Model(inputs=inputs, outputs=[hour_output, minute_output])

#Compile with separate losses for each output
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss={
        'hour_output': 'sparse_categorical_crossentropy',
        'minute_output': 'sparse_categorical_crossentropy'
    },
    loss_weights={
        'hour_output': 1.0,  
        'minute_output': 1.0
    },
    metrics={
        'hour_output': 'accuracy',
        'minute_output': 'accuracy'
    }
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
        'best_clock_multihead_model.keras',
        monitor='val_hour_output_accuracy',
        save_best_only=True,
        verbose=1,
        mode = 'max'
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
#Evaluate on test set
scores = model.evaluate(test_dataset, verbose=1)
print(f'\nTest Results:')
print(f'Overall loss: {scores[0]:.4f}')
print(f'Hour loss: {scores[1]:.4f}')
print(f'Minute loss: {scores[2]:.4f}')
print(f'Hour accuracy: {scores[3]:.4f} ({scores[3]*100:.2f}%)')
print(f'Minute accuracy: {scores[4]:.4f} ({scores[4]*100:.2f}%)')


#Get predictions
predictions = model.predict(test_dataset)
hour_predictions = predictions[0]
minute_predictions = predictions[1]

pred_hours = np.argmax(hour_predictions, axis=1)
pred_minutes = np.argmax(minute_predictions, axis=1)

#Get true labels from test set
test_labels_list = []
for _, (hour_labels_batch, minute_labels_batch) in test_dataset:
    test_labels_list.append((hour_labels_batch.numpy(), minute_labels_batch.numpy()))

true_hours = np.concatenate([h for h, m in test_labels_list], axis=0)
true_minutes = np.concatenate([m for h, m in test_labels_list], axis=0)

#Calculate exact match accuracy
exact_match = np.mean((pred_hours == true_hours) & (pred_minutes == true_minutes))
print(f'Exact match (both correct): {exact_match:.4f} ({exact_match*100:.2f}%)')

#Calculate common sense time difference accuracy
def calculate_time_difference_minutes(true_hours, true_minutes, pred_hours, pred_minutes):
    """
    Calculate absolute time difference in minutes, handling circular clock arithmetic.
    """
    #Convert to total minutes since midnight
    true_total_minutes = true_hours * 60 + true_minutes
    pred_total_minutes = pred_hours * 60 + pred_minutes
    
    #Calculate raw difference
    diff = np.abs(true_total_minutes - pred_total_minutes)
    
    #Handle wrap-around (e.g., 11:50 vs 00:10)
    diff = np.minimum(diff, 720 - diff)
    
    return diff

def format_time_difference(minutes):
    """Convert minutes to 'X hours Y minutes' format"""
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    return hours, mins

time_diffs = calculate_time_difference_minutes(true_hours, true_minutes, pred_hours, pred_minutes)

# Calculate average error in hours and minutes format
avg_error_hours, avg_error_mins = format_time_difference(time_diffs.mean())
median_error_hours, median_error_mins = format_time_difference(np.median(time_diffs))

print(f'\n"Common Sense" Time Difference Accuracy:')
print(f'Average error: {avg_error_hours} hour(s) and {avg_error_mins} minute(s)')
print(f'  (Total: {time_diffs.mean():.2f} minutes)')
print(f'Median error: {median_error_hours} hour(s) and {median_error_mins} minute(s)')
print(f'  (Total: {np.median(time_diffs):.2f} minutes)')
print(f'Std deviation: {time_diffs.std():.2f} minutes')
hour_errors = np.abs(pred_hours - true_hours)
hour_errors = np.minimum(hour_errors, 12 - hour_errors)  #Handle wraparound
minute_errors = np.abs(pred_minutes - true_minutes)
minute_errors = np.minimum(minute_errors, 60 - minute_errors)  #Handle wraparound
print(f'Average hour error: {hour_errors.mean():.2f} hours')
print(f'Average minute error: {minute_errors.mean():.2f} minutes')

def plot_learning_curves(history, save_path='learning_curves_multihead.png'):
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