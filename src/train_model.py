import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import os
import joblib

# Define the path to the root of the new dataset
data_dir = 'data/raw/Rock-Paper-Scissors'
model_dir = 'models'

def train_and_save_model():
    """
    Trains a Convolutional Neural Network (CNN) using the pre-split
    train and validation data from the new dataset. It then evaluates
    the final model on the test data and saves the model and class labels.
    """

    print("--- Starting model training... ---")

    # ImageDataGenerator for training data with data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,          # Normalize pixel values
        rotation_range=20,       # Randomly rotate images
        width_shift_range=0.2,   # Randomly shift images horizontally
        height_shift_range=0.2,  # Randomly shift images vertically
        shear_range=0.2,         # Shear transformations
        zoom_range=0.2,          # Randomly zoom into images
        horizontal_flip=True,    # Randomly flip images horizontally
        fill_mode='nearest',     # Strategy for filling in new pixels
    )

    # ImageDataGenerator for validation and test data (no augmentation)
    # We only rescale the pixel values
    val_test_datagen = ImageDataGenerator(rescale=1./255)

    # Create data generators for each set
    # --- The key change is here, pointing to the 'train' subdirectory! ---
    train_generator = train_datagen.flow_from_directory(
        os.path.join(data_dir, 'train'),
        target_size=(128, 128),  # All images will be resized to 128x128
        batch_size=32,
        class_mode='categorical',
    )

    validation_generator = val_test_datagen.flow_from_directory(
        os.path.join(data_dir, 'validation'),
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical',
    )
    
    test_generator = val_test_datagen.flow_from_directory(
        os.path.join(data_dir, 'test'),
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical',
        shuffle=False # Do not shuffle the test data
    )

    # Step 1: Define the CNN model architecture
    model = Sequential([
        # Convolutional Block 1
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D(2, 2),

        # Convolutional Block 2
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        # Convolutional Block 3
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        # Flatten the 3D feature maps to a 1D vector
        Flatten(),

        # A fully connected layer
        Dense(512, activation='relu'),
        Dropout(0.5), # Add dropout for regularization

        # Output layer with 3 units (for rock, paper, scissors)
        Dense(3, activation='softmax')
    ])

    # Step 2: Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Step 3: Train the model
    # Use an EarlyStopping callback to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    history = model.fit(
        train_generator,
        epochs=50,
        validation_data=validation_generator,
        callbacks=[early_stopping]
    )

    print("\n--- Model training complete ---")
    
    # Step 4: Evaluate the model on the test set
    print("\n--- Evaluating model performance on the test data ---")
    test_loss, test_acc = model.evaluate(test_generator, verbose=1)
    print(f"Final Test Accuracy: {test_acc*100:.2f}%")

    # Step 5: Save the trained model and class labels
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'rps_model.keras')
    class_labels_path = os.path.join(model_dir, 'class_labels.pkl')
    
    model.save(model_path)
    joblib.dump(train_generator.class_indices, class_labels_path)
    
    print(f"\nModel saved to: {model_path}")
    print(f"Class labels saved to: {class_labels_path}")

if __name__ == "__main__":
    train_and_save_model()
