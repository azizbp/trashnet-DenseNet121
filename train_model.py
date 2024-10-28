import os
import wandb
import zipfile
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.regularizers import l2 
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Activation
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print(f"TensorFlow version: {tf.__version__}")

# connect to wandb.ai
wandb.init(project="trashnet-model", entity="azizbp-gunadarma-university")

# configuration
config = {
    "learning_rate": 0.001,
    "epochs": 50,
    "batch_size": 32,
    "image_size": 224
}

wandb.config.update(config)

# unzip the dataset
zip_file_path = 'trashnet/dataset-resized.zip'
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall('trashnet')

# remove .DS_Store if exists
ds_store_path = 'trashnet/dataset-resized/.DS_Store'
if os.path.exists(ds_store_path):
    os.remove(ds_store_path)
    print(".DS_Store file removed successfully.")
else:
    print(".DS_Store file does not exist.")

# initialize directories
train_dir = 'trashnet/dataset-resized'
classes = [os.listdir(train_dir)]
print(classes)

# Print dataset info
for class_name in classes:
    class_path = os.path.join(train_dir, class_name)
    if os.path.exists(class_path):
        print(f"{class_name}: {len(os.listdir(class_path))} images")

# Calculate class weights
total_samples = sum([len(os.listdir(os.path.join(train_dir, class_name))) for class_name in classes])
class_weights = {
    i: total_samples / (len(classes) * len(os.listdir(os.path.join(train_dir, class_name))))
    for i, class_name in enumerate(classes)
}

print("\nClass weights:", class_weights)

# Data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=45,
    width_shift_range=0.3,
    height_shift_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    shear_range=0.3,
    zoom_range=0.3,
    brightness_range=[0.7, 1.3],
    fill_mode='nearest',
    validation_split=0.2
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

print("\nSetting up data generators...")

# Create generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(config["image_size"], config["image_size"]),
    batch_size=config["batch_size"],
    class_mode='categorical',
    shuffle=True,
    subset='training'
)

validation_generator = val_datagen.flow_from_directory(
    train_dir,
    target_size=(config["image_size"], config["image_size"]),
    batch_size=config["batch_size"],
    class_mode='categorical',
    shuffle=False,
    subset='validation'
)

print("\nCreating model...")

# Create model
base_model = DenseNet121(
    input_shape=(config["image_size"], config["image_size"], 3),
    include_top=False,
    weights='imagenet'
)

# Freeze early layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Build model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(256, kernel_regularizer=l2(0.01))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(len(classes), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile model
optimizer = Adam(learning_rate=config["learning_rate"])
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nModel compiled successfully")

# Callbacks
callbacks = [
    ReduceLROnPlateau(
        monitor='val_loss',
        patience=2,
        factor=0.2,
        min_lr=1e-7,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
]

print("\nStarting training...")

# Training
try:
    history = model.fit(
        train_generator,
        epochs=config["epochs"],
        validation_data=validation_generator,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\nTraining completed successfully")
    
    # Log metrics to wandb
    for epoch in range(len(history.history['accuracy'])):
        wandb.log({
            "epoch": epoch + 1,
            "accuracy": float(history.history['accuracy'][epoch]),
            "loss": float(history.history['loss'][epoch]),
            "val_accuracy": float(history.history['val_accuracy'][epoch]),
            "val_loss": float(history.history['val_loss'][epoch])
        })
    
    print("\nMetrics logged to wandb")
    
    # Save model
    model.save("bestModel-trashnet_v9-densenet121.h5")
    artifact = wandb.Artifact("trashnet-model", type="model")
    artifact.add_file("bestModel-trashnet_v9-densenet121.h5")
    wandb.log_artifact(artifact)
    
    print("\nModel saved successfully")

except Exception as e:
    print(f"\nAn error occurred during training: {str(e)}")
    print(f"Error type: {type(e)}")
    print(f"Error details: {e.__dict__}")
    raise

finally:
    wandb.finish()