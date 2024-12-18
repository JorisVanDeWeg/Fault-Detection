import tensorflow as tf
import os

# Step 1: Define the Model
def create_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # For binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Step 2: Train the Model (using dummy data for demonstration)
# Replace X_train and y_train with your actual training data
input_shape = 10  # Example input size
X_train = tf.random.normal((100, input_shape))  # Dummy data
y_train = tf.random.uniform((100,), maxval=2, dtype=tf.int32)

model = create_model(input_shape)
model.fit(X_train, y_train, epochs=10, batch_size=4, validation_split=0.2)

# Step 3: Convert the Model to TensorFlow Lite Format with Quantization
def convert_to_tflite(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Enables 8-bit quantization
    tflite_model = converter.convert()
    return tflite_model

tflite_model = convert_to_tflite(model)

# Step 4: Save and Check the Model Size
def save_and_check_model_size(tflite_model, file_path='model.tflite'):
    with open(file_path, 'wb') as f:
        f.write(tflite_model)
    
    model_size = os.path.getsize(file_path) / 1024  # Size in KB
    print(f"Model size: {model_size:.2f} KB")
    
    # Check if model size is within limit for Arduino Nano 33 BLE Sense
    if model_size < 200:
        print("Model size is within Arduino Nano 33 BLE Sense limits.")
    else:
        print("Model size exceeds Arduino Nano 33 BLE Sense limits. Consider simplifying the model.")

save_and_check_model_size(tflite_model)