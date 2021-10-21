import os
import argparse
import numpy as np
import tensorflow as tf

from tensorflow_model_optimization.quantization.keras import vitis_quantize
from tensorflow.keras import models, optimizers
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Activation

def train_nn():

    #Set Parameters
    BATCHSIZE=32
    EPOCHS=3
    LEARN_RATE=0.0001
    DECAY_RATE=1e-6

    mnist_dataset = tf.keras.datasets.mnist.load_data('mnist_data')
    (x_train, y_train), (x_test, y_test) = mnist_dataset

    x_train = (x_train/255.0)  
    x_test = (x_test/255.0)

    x_train = np.reshape(x_train, [-1, 28, 28, 1])
    x_test = np.reshape(x_test, [-1, 28, 28, 1])

    # one-hot encode the labels
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)

    # take 5000 images from train set to make a dataset for prediction
    x_valid = x_train[55000:]
    y_valid = y_train[55000:]

    # reduce train dataset to 55000 images
    y_train = y_train[:55000]
    x_train = x_train[:55000]

    inputs=Input(shape=(28, 28, 1))
    net=Conv2D(28, kernel_size=(3, 3))(inputs)
    net=Flatten()(net)
    net=Dense(10)(net)
    prediction=Activation('softmax')(net)

    model = models.Model(inputs=inputs, outputs=prediction)
    print(model.summary())

    model.compile(loss='categorical_crossentropy', 
            optimizer=optimizers.Adam(lr=LEARN_RATE),
            metrics=['accuracy']
            )

    model.fit(x_train,
        y_train,
        validation_data=(x_valid, y_valid)
        )

    #Evaluate Model Accracy
    scores = model.evaluate(x_test,y_test)

    print('Loss: %.3f' % scores[0])
    print('Accuracy: %.3f' % scores[1])

    # save weights, model architecture & optimizer to an HDF5 format file
    tf.keras.backend.set_learning_phase(0)
    model.save(os.path.join('./train','keras_trained_model.h5'))
    print ('######## Finished Training ########')

    float_model = tf.keras.models.load_model('./train/keras_trained_model.h5')
    print('######## Quantizing Float Model ########')
    quantizer = vitis_quantize.VitisQuantizer(float_model)
    quantized_model = quantizer.quantize_model(calib_dataset=(x_valid, y_valid))
    quantized_model.save('./train/quantized_model.h5')
    # print('######## Evaluating the Quantized Model ########')
    # with vitis_quantize.quantize_scope():
    #     test_quantized_model = tf.keras.models.load_model('./train/quantized_model.h5')
    # test_quantized_model.compile( loss='categorical_crossentropy', 
    #     metrics= ['accuracy'])
    # test_quantized_model.evaluate(testing_data, 
    #                         testing_label,
    #                         batch_size=BATCHSIZE)

    #vitis_quantize.VitisQuantizer.dump_model(quantized_model, (testing_data, testing_label), './dump')




def main():

    train_nn()

    

if __name__ ==  "__main__":
    main()