import tensorflow as tf
from tensorflow.keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, SGD, RMSprop

HEIGHT = 32
WIDTH = 32
DEPTH = 3
NUM_CLASSES = 10

def get_model(learning_rate, weight_decay, optimizer, momentum, size, mpi=False, hvd=False):

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(HEIGHT, WIDTH, DEPTH)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES))
    model.add(Activation('softmax'))

    if mpi:
        size = hvd.size()

    if optimizer.lower() == 'sgd':
        opt = SGD(lr=learning_rate * size, decay=weight_decay, momentum=momentum)
    elif optimizer.lower() == 'rmsprop':
        opt = RMSprop(lr=learning_rate * size, decay=weight_decay)
    else:
        opt = Adam(lr=learning_rate * size, decay=weight_decay)

    if mpi:
        opt = hvd.DistributedOptimizer(opt)

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    
    return model

