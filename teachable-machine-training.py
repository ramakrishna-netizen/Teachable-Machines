import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import shutil
import matplotlib.pyplot as plt

class TeachableMachineTrainer:
    def __init__(self, base_folder='teachable_machine_data'):
        self.base_folder = base_folder
        self.classes_folder = os.path.join(base_folder, 'classes')
        self.models_folder = os.path.join(base_folder, 'models')
        self.training_folder = os.path.join(base_folder, 'training_data')
        self.image_size = (224, 224)  # Standard size for MobileNetV2
        
        # Create necessary directories
        os.makedirs(self.classes_folder, exist_ok=True)
        os.makedirs(self.models_folder, exist_ok=True)
        os.makedirs(self.training_folder, exist_ok=True)
        
        # The current model and class mapping
        self.model = None
        self.class_names = {}
    
    def preprocess_images(self, augmentation=True, samples_per_image=5):
        """
        Preprocess images from class folders and prepare them for training
        """
        print("Preprocessing images...")
        
        # Clear the training folder
        if os.path.exists(self.training_folder):
            shutil.rmtree(self.training_folder)
        os.makedirs(self.training_folder)
        
        class_dirs = [d for d in os.listdir(self.classes_folder) 
                     if os.path.isdir(os.path.join(self.classes_folder, d))]
        
        if len(class_dirs) < 2:
            raise ValueError("Need at least 2 classes with images to train a model")
        
        # Process each class
        for idx, class_dir in enumerate(class_dirs):
            class_path = os.path.join(self.classes_folder, class_dir)
            class_name = class_dir  # Use directory name as class name
            
            # Create corresponding directory in training folder
            train_class_dir = os.path.join(self.training_folder, class_name)
            os.makedirs(train_class_dir, exist_ok=True)
            
            # Copy and preprocess original images
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if not image_files:
                print(f"Warning: No images found in class {class_name}")
                continue
            
            print(f"Processing class {class_name} with {len(image_files)} images")
            
            # Copy and preprocess original images
            for img_file in image_files:
                src_path = os.path.join(class_path, img_file)
                dst_path = os.path.join(train_class_dir, img_file)
                
                # Preprocess and save the image
                try:
                    img = Image.open(src_path)
                    img = img.convert('RGB')  # Ensure RGB format
                    img = img.resize(self.image_size)
                    img.save(dst_path)
                except Exception as e:
                    print(f"Error processing {src_path}: {e}")
            
            # Store class name mapping
            self.class_names[idx] = class_name
            
            # Perform data augmentation if requested
            if augmentation and len(image_files) > 0:
                self._augment_class_images(train_class_dir, samples_per_image)
        
        print(f"Preprocessing complete. Found {len(class_dirs)} classes.")
        return len(class_dirs)
    
    def _augment_class_images(self, class_folder, samples_per_image=5):
        """Generate augmented images to improve model training"""
        # Define augmentation parameters
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Process each image in the class folder
        image_files = [f for f in os.listdir(class_folder) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for filename in image_files:
            image_path = os.path.join(class_folder, filename)
            
            # Load and preprocess the image
            try:
                img = Image.open(image_path)
                img = img.convert('RGB')
                img = img.resize(self.image_size)
                img_array = np.array(img)
                img_array = img_array.reshape((1,) + img_array.shape)
                
                # Generate augmented images
                i = 0
                for batch in datagen.flow(img_array, batch_size=1):
                    augmented_path = os.path.join(class_folder, f"aug_{i}_{filename}")
                    augmented_img = Image.fromarray(batch[0].astype('uint8'))
                    augmented_img.save(augmented_path)
                    i += 1
                    if i >= samples_per_image:
                        break
            except Exception as e:
                print(f"Error augmenting image {image_path}: {e}")
    
    def build_model(self, num_classes):
        """Build the model architecture using MobileNetV2 as base"""
        print(f"Building model with {num_classes} classes...")
        
        # Use MobileNetV2 as the base model
        base_model = MobileNetV2(weights='imagenet', include_top=False, 
                                input_shape=(224, 224, 3))
        
        # Add custom classification head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu')(x)
        predictions = Dense(num_classes, activation='softmax')(x)
        
        # Create the full model
        model = Model(inputs=base_model.input, outputs=predictions)
        
        # Freeze the base model layers
        for layer in base_model.layers:
            layer.trainable = False
            
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train(self, epochs=10, batch_size=32, validation_split=0.2):
        """Train the model with the prepared data"""
        print("Starting model training...")
        
        # Make sure we have preprocessing done
        if not os.path.exists(self.training_folder) or len(os.listdir(self.training_folder)) < 2:
            num_classes = self.preprocess_images()
        else:
            num_classes = len(os.listdir(self.training_folder))
        
        # Build the model if not already built
        if self.model is None:
            self.build_model(num_classes)
        
        # Set up data generators
        datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            validation_split=validation_split
        )
        
        # Create train and validation generators
        train_generator = datagen.flow_from_directory(
            self.training_folder,
            target_size=self.image_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='training'
        )
        
        validation_generator = datagen.flow_from_directory(
            self.training_folder,
            target_size=self.image_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation'
        )
        
        # Update class names mapping from the generator
        self.class_indices = {v: k for k, v in train_generator.class_indices.items()}
        print(f"Class mapping: {self.class_indices}")
        
        # Define callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=2
            )
        ]
        
        # Train the model
        history = self.model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // batch_size,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // batch_size,
            epochs=epochs,
            callbacks=callbacks
        )
        
        # Save the trained model
        model_path = os.path.join(self.models_folder, "teachable_machine_model.h5")
        self.model.save(model_path)
        print(f"Model saved to {model_path}")
        
        # Save class indices
        import json
        with open(os.path.join(self.models_folder, "class_indices.json"), 'w') as f:
            json.dump(self.class_indices, f)
        
        # Plot training history
        self._plot_training_history(history)
        
        # Return validation accuracy
        val_acc = history.history['val_accuracy'][-1]
        print(f"Training complete. Validation accuracy: {val_acc:.4f}")
        return history, val_acc
    
    def _plot_training_history(self, history):
        """Plot and save training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_ylabel('Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.legend()
        
        # Plot loss
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_ylabel('Loss')
        ax2.set_xlabel('Epoch')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.models_folder, 'training_history.png'))
        plt.close()
    
    def export_model_tfjs(self):
        """Export the model to TensorFlow.js format"""
        if self.model is None:
            raise ValueError("No model trained yet. Call train() first.")
        
        # Create export directory
        export_dir = os.path.join(self.models_folder, "tfjs_model")
        os.makedirs(export_dir, exist_ok=True)
        
        # Export to TensorFlow.js format
        try:
            # Note: You would need to install the tensorflowjs package first
            # pip install tensorflowjs
            import tensorflowjs as tfjs
            tfjs.converters.save_keras_model(self.model, export_dir)
            print(f"Model exported to TensorFlow.js format at {export_dir}")
            
            # Copy class indices to export directory
            import json
            with open(os.path.join(self.models_folder, "class_indices.json"), 'r') as f:
                class_indices = json.load(f)
            
            with open(os.path.join(export_dir, "metadata.json"), 'w') as f:
                json.dump({
                    "labels": class_indices,
                    "imageSize": self.image_size
                }, f)
            
            return export_dir
        except ImportError:
            print("TensorFlow.js converter not found. Install with: pip install tensorflowjs")
            # Create a simulated export
            with open(os.path.join(export_dir, "model.json"), 'w') as f:
                f.write('{"format": "tfjs", "message": "This is a placeholder. Real export requires tensorflowjs package."}')
            return export_dir
    
    def predict(self, image_path):
        """Make a prediction with a single image"""
        if self.model is None:
            raise ValueError("No model trained yet. Call train() first.")
        
        try:
            # Preprocess the image
            img = Image.open(image_path)
            img = img.convert('RGB')
            img = img.resize(self.image_size)
            img_array = np.array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            
            # Make prediction
            predictions = self.model.predict(img_array)
            class_idx = np.argmax(predictions[0])
            confidence = predictions[0][class_idx]
            
            # Get class name
            class_name = self.class_indices.get(class_idx, f"Class_{class_idx}")
            
            return {
                "class_name": class_name,
                "class_idx": int(class_idx),
                "confidence": float(confidence),
                "predictions": {self.class_indices.get(i, f"Class_{i}"): float(p) 
                               for i, p in enumerate(predictions[0])}
            }
        except Exception as e:
            return {"error": str(e)}


# Example usage
if __name__ == "__main__":
    # Create a trainer instance
    trainer = TeachableMachineTrainer(base_folder="teachable_machine_data")
    
    # You would organize your images in class folders:
    # teachable_machine_data/classes/cat/[cat images]
    # teachable_machine_data/classes/dog/[dog images]
    # teachable_machine_data/classes/bird/[bird images]
    
    # Preprocess images (also happens automatically during training)
    trainer.preprocess_images(augmentation=True, samples_per_image=3)
    
    # Train the model
    history, accuracy = trainer.train(epochs=15, batch_size=32)
    
    # Export model
    export_dir = trainer.export_model_tfjs()
    
    # Make a prediction
    # result = trainer.predict("path_to_test_image.jpg")
    # print(result)
