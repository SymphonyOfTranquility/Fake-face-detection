import os
import glob
import pickle

import keras
from keras.layers import TimeDistributed, GRU, Dense, Dropout, Activation
import tensorflow

from keras_video import VideoFrameGenerator

classes = [i.split(os.path.sep)[1] for i in glob.glob('videos/*')]
classes.sort()
print(classes)

SIZE = (185, 185)
CHANNELS = 3
NBFRAME = 5
BATCH_SIZE = 8
glob_pattern= 'videos/{classname}/*.mp4'

data_aug = keras.preprocessing.image.ImageDataGenerator(
    zoom_range=.1,
    horizontal_flip=True,
    rotation_range=8,
    width_shift_range=.2,
    height_shift_range=.2
)

train = VideoFrameGenerator(
    classes=classes, 
    glob_pattern=glob_pattern,
    nb_frames=NBFRAME,
    split=.25, 
    shuffle=True,
    batch_size=BATCH_SIZE,
    target_shape=SIZE,
    nb_channel=CHANNELS,
    transformation=None,
    use_frame_cache=False)

valid = train.get_validation_generator()


def build_mobilenet(shape):
    model = keras.applications.mobilenet_v2.MobileNetV2(
        include_top=False,
        input_shape=shape,
        weights='imagenet'
    )
    output = keras.layers.GlobalMaxPool2D()
    return keras.Sequential([model, output])


def action_model(shape, num_classes):
    convnet = build_mobilenet(shape[1:])

    model = keras.Sequential()

    model.add(TimeDistributed(convnet, input_shape=shape))
    model.add(GRU(64))
    model.add(Dense(1024, activation='linear'))
    model.add(Activation('relu'))
    model.add(Dropout(.33))
    model.add(Dense(512, activation='linear'))
    model.add(Activation('relu'))
    model.add(Dropout(.33))
    model.add(Dense(128, activation='linear'))
    model.add(Activation('relu'))
    model.add(Dropout(.33))
    model.add(Dense(64, activation='linear'))
    model.add(Activation('relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model


INSHAPE=(NBFRAME,) + SIZE + (CHANNELS,)
model = action_model(INSHAPE, len(classes))
optimizer = tensorflow.keras.optimizers.SGD()
model.compile(
    optimizer,
    'categorical_crossentropy',
    metrics=['acc']
)

EPOCHS = 50
callbacks = [
    keras.callbacks.ReduceLROnPlateau(verbose=1),
    keras.callbacks.ModelCheckpoint(
        'lab_work/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
        verbose=1),
]

print(model.summary())
history = model.fit_generator(
    train,
    validation_data=valid,
    verbose=1,
    epochs=EPOCHS,
    callbacks=callbacks
)

with open('train_history', 'wb') as f:
    pickle.dump(history.history, f)

model.save('lab_model_final.h5')
