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

# connect to wandb.ai
wandb.init(project="trashnet-model", entity="azizbp-gunadarma-university")

# Configuration
config = {
    "learning_rate": 0.001,
    "epochs": 50,
    "batch_size": 32,
    "image_size": 224
}

wandb.config.update(config)

# unzip the dataset
zip_file_path = 'dataset-resized.zip'
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall('trashnet')

# remove .DS_Store if exists
ds_store_path = 'trashnet/dataset-resized/.DS_Store'
if os.path.exists(ds_store_path):
    os.remove(ds_store_path)

# Initialize directories
train_dir = 'trashnet/dataset-resized'
classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Calculate class weights
total_samples = sum([len(os.listdir(os.path.join(train_dir, class_name))) for class_name in classes])
class_weights = {
    i: total_samples / (len(classes) * len(os.listdir(os.path.join(train_dir, class_name))))
    for i, class_name in enumerate(classes)
}

# Data generators with proper dtype specification
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
    validation_split=0.2,
    dtype=tf.float32
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    dtype=tf.float32
)

# Create generators with proper setup
train_generator = train_datagen.flow_from_directory(
    directory=train_dir,
    target_size=(config["image_size"], config["image_size"]),
    batch_size=config["batch_size"],
    class_mode='categorical',
    shuffle=True,
    subset='training',
    classes=classes
)

validation_generator = val_datagen.flow_from_directory(
    directory=train_dir,
    target_size=(config["image_size"], config["image_size"]),
    batch_size=config["batch_size"],
    class_mode='categorical',
    shuffle=False,
    subset='validation',
    classes=classes
)

# Create model with proper input specification
input_shape = (config["image_size"], config["image_size"], 3)
base_model = DenseNet121(
    input_shape=input_shape,
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

# Compile with mixed precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')
optimizer = Adam(learning_rate=config["learning_rate"])
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
lr_reduction = ReduceLROnPlateau(
    monitor='val_loss',
    patience=2,
    factor=0.2,
    min_lr=1e-7,
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy', 0) > 0.9) and (logs.get('val_accuracy', 0) > 0.9):
            self.model.stop_training = True

# Training
try:
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=config["epochs"],
        class_weight=class_weights,
        workers=4,
        use_multiprocessing=True,
        callbacks=[MyCallback(), early_stopping, lr_reduction]
    )
    
    # Log metrics to wandb
    for epoch in range(len(history.history['accuracy'])):
        wandb.log({
            "epoch": epoch + 1,
            "accuracy": history.history['accuracy'][epoch],
            "loss": history.history['loss'][epoch],
            "val_accuracy": history.history['val_accuracy'][epoch],
            "val_loss": history.history['val_loss'][epoch]
        })
    
    # Save model
    model.save("bestModel-trashnet_v9-densenet121.h5")
    artifact = wandb.Artifact("trashnet-model", type="model")
    artifact.add_file("bestModel-trashnet_v9-densenet121.h5")
    wandb.log_artifact(artifact)

except Exception as e:
    print(f"An error occurred during training: {str(e)}")
    raise

finally:
    wandb.finish()