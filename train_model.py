# import library
import os
import wandb
import zipfile
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.regularizers import l2 
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten, BatchNormalization, Activation
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# connect to wandb.ai
wandb.init(project="trashnet-model", entity="azizbp-gunadarma-university")

# unzip the dataset
zip_file_path = 'trashnet/dataset-resized.zip'

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall('trashnet')

os.listdir('trashnet/dataset-resized')

# remove the .DS_Store file
ds_store_path = 'trashnet/dataset-resized/.DS_Store'

if os.path.exists(ds_store_path):
    os.remove(ds_store_path)
    print(".DS_Store file removed successfully.")
else:
    print(".DS_Store file does not exist.")

# intialize the directory and see classes
train_dir = 'trashnet/dataset-resized'
classes = os.listdir(train_dir)

# using class weights
total_samples = sum([len(os.listdir(os.path.join(train_dir, class_name))) 
                    for class_name in ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']])
class_weights = {
    0: total_samples / (6 * len(os.listdir(os.path.join(train_dir, 'cardboard')))),
    1: total_samples / (6 * len(os.listdir(os.path.join(train_dir, 'glass')))),
    2: total_samples / (6 * len(os.listdir(os.path.join(train_dir, 'metal')))),
    3: total_samples / (6 * len(os.listdir(os.path.join(train_dir, 'paper')))),
    4: total_samples / (6 * len(os.listdir(os.path.join(train_dir, 'plastic')))),
    5: total_samples / (6 * len(os.listdir(os.path.join(train_dir, 'trash'))))
}

# augmentation
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

# generator
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=True,
        subset='training'
)

validation_generator = val_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False,
        subset='validation'
)

# base model DenseNet121
base_model = DenseNet121(input_shape=(224, 224, 3),
                         include_top=False,
                         weights='imagenet')

for layer in base_model.layers[:-30]:
    layer.trainable = False

# several additions to layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(256, kernel_regularizer=l2(0.01))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(6, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# compile model
optimizer = Adam(learning_rate=0.001)

model.compile(loss = 'categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# setup config epoch, lr and batch size
config = {
    "learning_rate": 0.001,
    "epochs": 50,
    "batch_size": 32
}

wandb.config.update(config)

# try to reduction the lr
lr_reduction = ReduceLROnPlateau(monitor='val_loss', 
                                 patience=2, 
                                 factor=0.2,
                                 min_lr=1e-7,
                                 verbose=1)

# early stopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# define callback
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > 0.9) and (logs.get('val_accuracy') > 0.9) :
            self.model.stop_training = True

callbacks = [myCallback(), early_stopping, lr_reduction]

# begin training
history = model.fit(train_generator, 
                    validation_data=validation_generator, 
                    batch_size=["batch_size"], 
                    epochs=config["epochs"],
                    class_weight=class_weights,
                    verbose=1,
                    callbacks=callbacks
                    )

# save epoch in wandb
for epoch in range(config["epochs"]):

    wandb.log({
        "epoch": epoch + 1,
        "accuracy": 'accuracy',
        "loss": 'loss',
        "val_accuracy": 'val_accuracy',
        "val_loss": 'val_loss'
    })


artifact = wandb.Artifact("trashnet-model", type="model")
artifact.add_file("bestModel-trashnet_v9-densenet121.h5")
wandb.log_artifact(artifact)

wandb.finish()