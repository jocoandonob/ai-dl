def get_code_examples():
    """
    Return a dictionary of code examples for various neural network implementations.
    
    Returns:
    --------
    code_examples : dict
        Dictionary of code examples
    """
    code_examples = {}
    
    # Simple Feedforward Neural Network
    code_examples["feedforward"] = """
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate synthetic data for binary classification
def generate_data(n_samples=1000):
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0]**2 + X[:, 1]**2 < 1).astype(int)
    return X, y

# Generate and split data
X, y = generate_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build model
model = keras.Sequential([
    keras.layers.Input(shape=(2,)),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train model
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Evaluate model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

# Make predictions
predictions = model.predict(X_test)
predicted_classes = (predictions > 0.5).astype(int).flatten()
"""

    # Deep Neural Network
    code_examples["deep_nn"] = """
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load and prepare data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Preprocess data
x_train = x_train.reshape(-1, 28*28).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28*28).astype('float32') / 255.0

# One-hot encode labels
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Build deep neural network
model = keras.Sequential([
    keras.layers.Input(shape=(784,)),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Add early stopping and learning rate reduction
callbacks = [
    keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2)
]

# Train model
history = model.fit(
    x_train, y_train,
    epochs=20,
    batch_size=128,
    validation_split=0.1,
    callbacks=callbacks,
    verbose=1
)

# Evaluate model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")
"""

    # Custom Training Loop
    code_examples["custom_training"] = """
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate synthetic data
np.random.seed(42)
X = np.random.randn(1000, 20)
y = (X[:, 0] + X[:, 1] * 0.5 + np.random.randn(1000) * 0.1 > 0).astype(np.float32)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build model
model = keras.Sequential([
    keras.layers.Input(shape=(20,)),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# Define loss function and optimizer
loss_fn = keras.losses.BinaryCrossentropy()
optimizer = keras.optimizers.Adam(learning_rate=0.001)
train_acc_metric = keras.metrics.BinaryAccuracy()
val_acc_metric = keras.metrics.BinaryAccuracy()

# Convert data to tf.data.Dataset
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

val_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
val_dataset = val_dataset.batch(64)

# Define training step
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_fn(y, logits)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    train_acc_metric.update_state(y, logits)
    return loss_value

# Define validation step
@tf.function
def test_step(x, y):
    val_logits = model(x, training=False)
    val_acc_metric.update_state(y, val_logits)

# Training loop
epochs = 20
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    
    # Training phase
    train_acc_metric.reset_states()
    train_losses = []
    
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        loss_value = train_step(x_batch_train, y_batch_train)
        train_losses.append(float(loss_value))
        
        if step % 50 == 0:
            print(f"Step {step}: loss = {float(loss_value):.4f}")
    
    train_acc = train_acc_metric.result()
    print(f"Training accuracy: {float(train_acc):.4f}")
    
    # Validation phase
    val_acc_metric.reset_states()
    
    for x_batch_val, y_batch_val in val_dataset:
        test_step(x_batch_val, y_batch_val)
    
    val_acc = val_acc_metric.result()
    print(f"Validation accuracy: {float(val_acc):.4f}")
"""

    # CNN Example
    code_examples["cnn"] = """
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Preprocess data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# One-hot encode labels
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Define data augmentation
data_augmentation = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.1),
    keras.layers.RandomZoom(0.1),
])

# Build CNN model
model = keras.Sequential([
    # Data augmentation (only applied during training)
    data_augmentation,
    
    # First convolutional block
    keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3)),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.Conv2D(32, (3, 3), padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.2),
    
    # Second convolutional block
    keras.layers.Conv2D(64, (3, 3), padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.Conv2D(64, (3, 3), padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.3),
    
    # Third convolutional block
    keras.layers.Conv2D(128, (3, 3), padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.Conv2D(128, (3, 3), padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.4),
    
    # Fully connected layers
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation='softmax')
])

# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Define callbacks
callbacks = [
    keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=5),
]

# Train model
history = model.fit(
    x_train, y_train,
    epochs=50,
    batch_size=64,
    validation_split=0.1,
    callbacks=callbacks,
    verbose=1
)

# Evaluate model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")
"""

    # RNN Example
    code_examples["rnn"] = """
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Generate synthetic time series data
def generate_time_series(n_steps=1000):
    time = np.arange(0, n_steps)
    # Trend component
    trend = 0.001 * time
    # Seasonal component
    seasonal = 0.5 * np.sin(2 * np.pi * time / 50)
    # Noise component
    noise = 0.1 * np.random.randn(n_steps)
    # Combine components
    series = trend + seasonal + noise
    return series

# Generate data
series = generate_time_series()

# Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(series.reshape(-1, 1))

# Create sequences
def create_sequences(data, seq_length=10):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Create sequence data
seq_length = 20
X, y = create_sequences(scaled_data, seq_length)

# Split into train and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build LSTM model
model = keras.Sequential([
    keras.layers.LSTM(50, activation='relu', input_shape=(seq_length, 1), return_sequences=True),
    keras.layers.Dropout(0.2),
    keras.layers.LSTM(50, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1)
])

# Compile model
model.compile(optimizer='adam', loss='mse')

# Train model
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# Make predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Inverse transform predictions
train_predictions = scaler.inverse_transform(train_predictions)
y_train_inv = scaler.inverse_transform(y_train)
test_predictions = scaler.inverse_transform(test_predictions)
y_test_inv = scaler.inverse_transform(y_test)

# Calculate RMSE
from sklearn.metrics import mean_squared_error
train_rmse = np.sqrt(mean_squared_error(y_train_inv, train_predictions))
test_rmse = np.sqrt(mean_squared_error(y_test_inv, test_predictions))
print(f"Train RMSE: {train_rmse:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(scaler.inverse_transform(scaled_data), label='Actual Data')

# Shift train predictions for plotting
train_plot = np.empty_like(scaled_data)
train_plot[:] = np.nan
train_plot[seq_length:len(train_predictions)+seq_length] = train_predictions

# Shift test predictions for plotting
test_plot = np.empty_like(scaled_data)
test_plot[:] = np.nan
test_plot[len(train_predictions)+seq_length:len(scaled_data)] = test_predictions

plt.plot(train_plot, label='Train Predictions')
plt.plot(test_plot, label='Test Predictions')
plt.legend()
plt.title('Time Series Prediction with LSTM')
plt.show()
"""

    # Transfer Learning Example
    code_examples["transfer_learning"] = """
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define image size
img_height, img_width = 224, 224
batch_size = 32

# Set up data generators with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

# Load data from directories
train_generator = train_datagen.flow_from_directory(
    'train_dir',  # replace with your directory
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    'train_dir',  # replace with your directory
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Get class names and number of classes
class_names = list(train_generator.class_indices.keys())
num_classes = len(class_names)

# Load pre-trained model
base_model = ResNet50(
    weights='imagenet', 
    include_top=False, 
    input_shape=(img_height, img_width, 3)
)

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers
model = keras.Sequential([
    base_model,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(num_classes, activation='softmax')
])

# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

# Fine-tuning: unfreeze some layers and train with lower learning rate
# Unfreeze the last 15 layers
for layer in model.layers[0].layers[-15:]:
    layer.trainable = True

# Recompile with lower learning rate
model.compile(
    optimizer=keras.optimizers.Adam(1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Continue training
history_fine = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

# Save model
model.save('fine_tuned_resnet50.h5')
"""

    # Object Detection Example
    code_examples["object_detection"] = """
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import time

# Load pre-trained model from TensorFlow Hub
detector = hub.load("https://tfhub.dev/tensorflow/efficientdet/d0/1")

# Function to run inference on a single image
def detect_objects(image_path):
    # Load and preprocess image
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    
    # Convert to float and add batch dimension
    input_img = tf.cast(img, tf.float32)
    input_img = input_img[tf.newaxis, ...]
    
    # Run detection
    start_time = time.time()
    result = detector(input_img)
    end_time = time.time()
    
    # Process result
    result = {key: value.numpy() for key, value in result.items()}
    
    # Get information about detection
    boxes = result["detection_boxes"][0]
    classes = result["detection_classes"][0].astype(np.int32)
    scores = result["detection_scores"][0]
    
    # Load COCO class names
    with open('coco_labels.txt', 'r') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    
    # Convert image to numpy for visualization
    image_np = img.numpy().astype(np.uint8)
    h, w, _ = image_np.shape
    
    # Draw boxes and labels on the image
    for i in range(min(10, len(scores))):
        if scores[i] >= 0.5:  # Confidence threshold
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * w, xmax * w, ymin * h, ymax * h)
            
            # Draw bounding box
            cv2.rectangle(
                image_np,
                (int(left), int(top)),
                (int(right), int(bottom)),
                (0, 255, 0),
                2
            )
            
            # Draw label
            class_name = class_names[classes[i]]
            label = f"{class_name}: {scores[i]:.2f}"
            cv2.putText(
                image_np,
                label,
                (int(left), int(top) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
    
    print(f"Inference time: {(end_time - start_time)*1000:.2f} ms")
    
    return image_np

# Example usage:
# detected_image = detect_objects("path/to/your/image.jpg")
# plt.figure(figsize=(12, 12))
# plt.imshow(detected_image)
# plt.show()
"""

    # Text Classification Example
    code_examples["text_classification"] = """
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import re
import string

# Load IMDB dataset
max_features = 10000
max_len = 200

(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=max_features)

# Pad sequences to the same length
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_len)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_len)

# Define vocabulary size and embedding dimension
vocab_size = max_features
embedding_dim = 128

# Build model
model = keras.Sequential([
    keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_len),
    keras.layers.Conv1D(128, 5, activation='relu'),
    keras.layers.GlobalMaxPooling1D(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation='sigmoid')
])

# Alternative model with LSTM
model_lstm = keras.Sequential([
    keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_len),
    keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True)),
    keras.layers.Bidirectional(keras.layers.LSTM(32)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation='sigmoid')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train model
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Evaluate model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

# Function to preprocess text for prediction
def preprocess_text(text):
    # Get word index from the dataset
    word_index = keras.datasets.imdb.get_word_index()
    
    # Preprocess text
    text = text.lower()
    text = re.sub(f'[{string.punctuation}]', ' ', text)
    
    words = text.split()
    sequence = []
    
    # Map words to indices
    for word in words:
        if word in word_index and word_index[word] < max_features:
            sequence.append(word_index[word] + 3)  # Adding 3 as offset (0 = padding, 1 = start, 2 = unknown)
    
    # Pad sequence
    if len(sequence) > max_len:
        sequence = sequence[:max_len]
    else:
        sequence = [0] * (max_len - len(sequence)) + sequence
    
    return np.array([sequence])

# Function to predict sentiment
def predict_sentiment(text):
    preprocessed = preprocess_text(text)
    prediction = model.predict(preprocessed)[0][0]
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    
    return sentiment, confidence

# Example usage:
# review = "This movie was fantastic! I really enjoyed the plot and acting."
# sentiment, confidence = predict_sentiment(review)
# print(f"Sentiment: {sentiment}, Confidence: {confidence:.2f}")
"""

    # Word Embeddings Example
    code_examples["word_embeddings"] = """
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import io
import os
from sklearn.manifold import TSNE

# Load dataset with texts
max_features = 20000  # Maximum number of words in the vocabulary
maxlen = 100  # Maximum sequence length

(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=max_features)

# Get word index
word_index = keras.datasets.imdb.get_word_index()

# Create reverse mapping
reverse_word_index = {value: key for key, value in word_index.items()}
def decode_review(text):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in text])

# Pad sequences
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

# Define model with embedding layer
embedding_dim = 100

model = keras.Sequential([
    keras.layers.Embedding(max_features, embedding_dim, input_length=maxlen),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train model
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.2,
    verbose=1
)

# Extract embeddings
embedding_layer = model.layers[0]
weights = embedding_layer.get_weights()[0]

# Function to get word vector
def get_word_vector(word):
    # Get word index
    idx = word_index.get(word)
    if idx is None or idx >= max_features:
        return None
    return weights[idx]

# Function to find most similar words
def find_similar_words(word, top_k=10):
    word_vec = get_word_vector(word)
    if word_vec is None:
        return []
    
    # Compute similarities
    similarities = np.dot(weights, word_vec) / (
        np.linalg.norm(weights, axis=1) * np.linalg.norm(word_vec)
    )
    
    # Get top k similar words
    most_similar = np.argsort(similarities)[-top_k-1:-1][::-1]
    
    # Convert indices back to words
    return [(reverse_word_index.get(idx - 3, '?'), similarities[idx]) 
            for idx in most_similar]

# Visualize embeddings with t-SNE
def plot_embeddings(words):
    # Get word vectors
    word_vecs = []
    valid_words = []
    
    for word in words:
        vec = get_word_vector(word)
        if vec is not None:
            word_vecs.append(vec)
            valid_words.append(word)
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    reduced_vecs = tsne.fit_transform(word_vecs)
    
    # Plot results
    plt.figure(figsize=(12, 10))
    
    plt.scatter(reduced_vecs[:, 0], reduced_vecs[:, 1], color='blue', alpha=0.5)
    
    for i, word in enumerate(valid_words):
        plt.annotate(word, reduced_vecs[i], fontsize=12)
    
    plt.title("t-SNE visualization of word embeddings")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

# Example usage:
# common_words = ["good", "bad", "movie", "film", "actor", "story", "character", 
#                "great", "terrible", "excellent", "poor", "amazing", "awful"]
# plot_embeddings(common_words)
# 
# similar_words = find_similar_words("excellent", top_k=5)
# print("Words similar to 'excellent':")
# for word, similarity in similar_words:
#     print(f"{word}: {similarity:.4f}")
"""

    # Data Preprocessing Example
    code_examples["data_preprocessing"] = """
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Load sample data (or create synthetic data)
def create_synthetic_data(n_samples=1000):
    np.random.seed(42)
    
    # Create features
    age = np.random.normal(40, 10, n_samples)
    income = 20000 + age * 500 + np.random.normal(0, 10000, n_samples)
    gender = np.random.choice(['Male', 'Female'], n_samples)
    education = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples)
    
    # Create target based on features
    purchase_prob = 1 / (1 + np.exp(-(age / 100 + income / 100000 - 2)))
    purchased = np.random.binomial(1, purchase_prob)
    
    # Add some missing values
    age[np.random.choice(n_samples, 50)] = np.nan
    income[np.random.choice(n_samples, 50)] = np.nan
    gender[np.random.choice(n_samples, 50)] = None
    education[np.random.choice(n_samples, 50)] = None
    
    # Create dataframe
    data = pd.DataFrame({
        'Age': age,
        'Income': income,
        'Gender': gender,
        'Education': education,
        'Purchased': purchased
    })
    
    return data

# Get data
data = create_synthetic_data()

# Split features and target
X = data.drop('Purchased', axis=1)
y = data['Purchased']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing for numerical and categorical columns
numerical_features = ['Age', 'Income']
categorical_features = ['Gender', 'Education']

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create and preprocess the training and test data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Check the shape of processed data
print(f"Processed training data shape: {X_train_processed.shape}")

# Get feature names after one-hot encoding
cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
cat_features = categorical_features
cat_columns = []

for i, col in enumerate(cat_features):
    cat_columns.extend([f"{col}_{val}" for val in cat_encoder.categories_[i]])

feature_names = numerical_features + cat_columns
print("Feature names after preprocessing:", feature_names)

# Build neural network model
model = keras.Sequential([
    keras.layers.Input(shape=(X_train_processed.shape[1],)),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train model
history = model.fit(
    X_train_processed, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Evaluate model
test_loss, test_acc = model.evaluate(X_test_processed, y_test)
print(f"Test accuracy: {test_acc:.4f}")

# Define a function for preprocessing new data
def preprocess_new_data(df):
    return preprocessor.transform(df)

# Example of preprocessing new data
new_data = pd.DataFrame({
    'Age': [35, 42, 28],
    'Income': [60000, 80000, 45000],
    'Gender': ['Male', 'Female', 'Male'],
    'Education': ['Bachelor', 'Master', 'High School']
})

processed_new_data = preprocess_new_data(new_data)
predictions = model.predict(processed_new_data)
print("Predictions for new data:", predictions)
"""

    # Model Evaluation Example
    code_examples["model_evaluation"] = """
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, 
    auc, precision_recall_curve, average_precision_score
)
import seaborn as sns

# Load sample dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Preprocess data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Convert to one-hot encoding
num_classes = 10
y_train_onehot = keras.utils.to_categorical(y_train, num_classes)
y_test_onehot = keras.utils.to_categorical(y_test, num_classes)

# Build model
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(num_classes, activation='softmax')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
history = model.fit(
    x_train, y_train_onehot,
    epochs=5,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)

# Get model predictions
y_pred_prob = model.predict(x_test)
y_pred = np.argmax(y_pred_prob, axis=1)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# ROC Curve and AUC for each class
plt.figure(figsize=(10, 8))

for i in range(num_classes):
    fpr, tpr, _ = roc_curve(y_test_onehot[:, i], y_pred_prob[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'Class {i} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True, linestyle='--', alpha=0.3)
plt.show()

# Precision-Recall curve for each class
plt.figure(figsize=(10, 8))

for i in range(num_classes):
    precision, recall, _ = precision_recall_curve(y_test_onehot[:, i], y_pred_prob[:, i])
    ap = average_precision_score(y_test_onehot[:, i], y_pred_prob[:, i])
    plt.plot(recall, precision, lw=2, label=f'Class {i} (AP = {ap:.2f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="best")
plt.grid(True, linestyle='--', alpha=0.3)
plt.show()

# Error Analysis - Find misclassified examples
misclassified_indices = np.where(y_pred != y_test)[0]
num_examples = min(10, len(misclassified_indices))

plt.figure(figsize=(15, 10))
for i, idx in enumerate(misclassified_indices[:num_examples]):
    plt.subplot(2, 5, i + 1)
    img = x_test[idx].reshape(28, 28)
    plt.imshow(img, cmap='gray')
    plt.title(f"True: {y_test[idx]}, Pred: {y_pred[idx]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# Training and validation curves
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()

# Per-class accuracy
class_accuracy = np.zeros(num_classes)
for i in range(num_classes):
    class_mask = (y_test == i)
    class_accuracy[i] = np.mean(y_pred[class_mask] == i)

plt.figure(figsize=(10, 6))
plt.bar(range(num_classes), class_accuracy)
plt.xlabel('Class')
plt.ylabel('Accuracy')
plt.title('Per-Class Accuracy')
plt.xticks(range(num_classes))
plt.ylim([0, 1])
plt.grid(True, linestyle='--', alpha=0.3, axis='y')
plt.show()
"""

    # Hyperparameter Tuning Example
    code_examples["hyperparameter_tuning"] = """
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
import itertools
import time

# Load sample data
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

# Preprocess data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Convert to one-hot encoding
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Define model building function
def build_model(learning_rate, num_conv_layers, num_filters, dropout_rate, num_dense_units):
    model = keras.Sequential()
    
    # Add convolutional layers
    for i in range(num_conv_layers):
        if i == 0:
            model.add(keras.layers.Conv2D(
                num_filters, (3, 3), activation='relu', padding='same',
                input_shape=(28, 28, 1)
            ))
        else:
            model.add(keras.layers.Conv2D(
                num_filters * (2 ** i), (3, 3), activation='relu', padding='same'
            ))
        model.add(keras.layers.MaxPooling2D((2, 2)))
    
    # Add flatten and dense layers
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(num_dense_units, activation='relu'))
    model.add(keras.layers.Dropout(dropout_rate))
    model.add(keras.layers.Dense(num_classes, activation='softmax'))
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Define parameter grid
param_grid = {
    'learning_rate': [0.001, 0.01],
    'num_conv_layers': [1, 2],
    'num_filters': [32, 64],
    'dropout_rate': [0.3, 0.5],
    'num_dense_units': [64, 128]
}

# Generate all parameter combinations
param_combinations = list(itertools.product(*param_grid.values()))
param_keys = list(param_grid.keys())

print(f"Total combinations to evaluate: {len(param_combinations)}")

# Grid search with cross-validation
def grid_search_cv(param_combinations, param_keys, X, y, n_splits=3):
    results = []
    
    for i, params in enumerate(param_combinations):
        param_dict = {k: v for k, v in zip(param_keys, params)}
        print(f"\nEvaluating combination {i+1}/{len(param_combinations)}:")
        for k, v in param_dict.items():
            print(f"{k}: {v}")
        
        # Create KFold
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_val_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            print(f"Fold {fold+1}/{n_splits}")
            
            # Split data
            X_fold_train, X_fold_val = X[train_idx], X[val_idx]
            y_fold_train, y_fold_val = y[train_idx], y[val_idx]
            
            # Build and train model
            model = build_model(**param_dict)
            start_time = time.time()
            history = model.fit(
                X_fold_train, y_fold_train,
                epochs=5,
                batch_size=64,
                validation_data=(X_fold_val, y_fold_val),
                verbose=0
            )
            train_time = time.time() - start_time
            
            # Get best validation score
            best_val_acc = max(history.history['val_accuracy'])
            fold_val_scores.append(best_val_acc)
            
            print(f"  Validation accuracy: {best_val_acc:.4f}, Training time: {train_time:.2f}s")
            
            # Clear session to free memory
            keras.backend.clear_session()
        
        # Calculate average validation score
        mean_val_score = np.mean(fold_val_scores)
        std_val_score = np.std(fold_val_scores)
        
        print(f"Mean validation accuracy: {mean_val_score:.4f} ± {std_val_score:.4f}")
        
        # Save results
        results.append({
            'params': param_dict,
            'mean_val_accuracy': mean_val_score,
            'std_val_accuracy': std_val_score
        })
    
    return results

# Use a subset of data for hyperparameter tuning
X_sample, _, y_sample, _ = train_test_split(
    x_train, y_train, test_size=0.8, random_state=42
)

# Perform grid search
results = grid_search_cv(param_combinations, param_keys, X_sample, y_sample)

# Find best parameters
best_idx = np.argmax([r['mean_val_accuracy'] for r in results])
best_params = results[best_idx]['params']
best_val_acc = results[best_idx]['mean_val_accuracy']

print("\nBest hyperparameters:")
for k, v in best_params.items():
    print(f"{k}: {v}")
print(f"Best validation accuracy: {best_val_acc:.4f}")

# Train final model with best parameters
final_model = build_model(**best_params)
final_history = final_model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

# Evaluate final model
test_loss, test_acc = final_model.evaluate(x_test, y_test)
print(f"Final test accuracy: {test_acc:.4f}")

# Plot hyperparameter importance
plt.figure(figsize=(12, 8))

for i, param in enumerate(param_keys):
    plt.subplot(2, 3, i+1)
    
    # Extract param values and corresponding accuracies
    param_values = [r['params'][param] for r in results]
    accuracies = [r['mean_val_accuracy'] for r in results]
    
    # Group by parameter value
    unique_values = sorted(set(param_values))
    grouped_accs = [
        [acc for pv, acc in zip(param_values, accuracies) if pv == val]
        for val in unique_values
    ]
    
    # Calculate mean accuracy for each parameter value
    mean_accs = [np.mean(group) for group in grouped_accs]
    
    # Plot
    plt.bar([str(val) for val in unique_values], mean_accs)
    plt.xlabel(param)
    plt.ylabel('Mean Validation Accuracy')
    plt.title(f'Effect of {param}')
    plt.ylim([min(mean_accs) - 0.05, max(mean_accs) + 0.05])
    plt.grid(True, linestyle='--', alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# Learning curves of final model
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(final_history.history['accuracy'], label='Training Accuracy')
plt.plot(final_history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(final_history.history['loss'], label='Training Loss')
plt.plot(final_history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()
"""
    
    return code_examples
