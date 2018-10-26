'''
Copyright (C) 2018  Sanchayan Santra

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''

# this will take the data and train the model

from keras.layers import Input, Conv2D, concatenate, Flatten, Dense
from keras.models import Model
import numpy as np


def make_model(inp_shape):
    # make the model
    inp1 = Input(shape=inp_shape, name='input1')
    inp2 = Input(shape=inp_shape, name='input2')

    conv1 = Conv2D(64, (3, 3), activation='tanh', padding='valid')
    conv2 = Conv2D(32, (3, 3), activation='tanh', padding='valid')
    conv3 = Conv2D(16, (3, 3), activation='tanh', padding='valid')

    x1 = conv1(inp1)
    x1 = conv2(x1)
    x1 = conv3(x1)

    x2 = conv1(inp2)
    x2 = conv2(x2)
    x2 = conv3(x2)

    x = concatenate([x1, x2], axis=1)
    x = Flatten()(x)
    x = Dense(16, activation='tanh')(x)
    x = Dense(8, activation='tanh')(x)
    output = Dense(2, activation='softmax', name='output')(x)

    model = Model(inputs=[inp1, inp2], outputs=output)

    return model


if __name__ == '__main__':
    # paramters
    num_epoch = 50
    batch_size = 500

    patches_f = './p_c_tpartition_30comp_2A.npy'
    labels_f = './p_c_tpartition_30comp_2A_labels.npy'
    model_file = 'model/comp_c_tpartition_30comp_2As.h5'

    # load the data
    print "loading data"
    patches = np.load(patches_f)
    labels = np.load(labels_f)
    print "loading done"

    [patch_types, numpatch, nch, ncol, nrow] = patches.shape
    assert patch_types == 2

    percent_train = 0.9
    numtrain = int(np.floor(numpatch*percent_train))
    numval = numpatch - numtrain

    inp_shape = [nch, ncol, nrow]

    # may take percent_train of the data in random way
    X_train1 = patches[0, :numtrain, :, :, :]
    X_train2 = patches[1, :numtrain, :, :, :]
    Y_train = labels[:numtrain]
    X_val1 = patches[0, numtrain:, :, :, :]
    X_val2 = patches[1, numtrain:, :, :, :]
    Y_val = labels[numtrain:]

    model = make_model(inp_shape)
    model.summary()

    model.compile(optimizer='adadelta', loss='mse', metrics=['accuracy'])

    model.fit([X_train1, X_train2], Y_train, epochs=num_epoch,
              verbose=1, batch_size=batch_size, shuffle=True,
              validation_data=([X_val1, X_val2], Y_val))

    model.save(model_file)
