#delighted by keras module
from __future__ import division
from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten,
    concatenate
)
from keras.layers.convolutional import (
    Conv2D,
    MaxPooling2D,
    AveragePooling2D
)
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K
##train import
import scipy,scipy.io
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, ModelCheckpoint
import numpy.random
import numpy as np
import pdb
from keras import optimizers
import h5py

##parameters
global ROW_AXIS 
global COL_AXIS 
global CHANNEL_AXIS
repetitions = [3,4,6,3]
type = 34 
batch_size = 100
nb_classes = 17
nb_epoch = 20
ROW_AXIS = 1
COL_AXIS = 2
CHANNEL_AXIS = 3

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-5)
early_stopper = EarlyStopping(min_delta=0.001, patience=20)
csv_logger = CSVLogger('resnet50_base.csv')
#checkpoint  = ModelCheckpoint('./check_file.h5', monitor='val_loss',
#                              verbose=0, save_best_only=False,
#                            save_weights_only=False, mode='auto', period=1)

#Resnet funcs
def _bn_relu(input):
    norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
    return Activation("relu")(norm)

def _bn_relu_conv(input,filters, kernel_size,
                      strides=(1,1), padding="same",
                      kernel_initializer="he_normal",
                      kernel_regularizer=l2(1.e-4)):

    activation = _bn_relu(input)
    return Conv2D(filters=filters, kernel_size=kernel_size,
                  strides=strides, padding=padding,
                  kernel_initializer=kernel_initializer,
                  kernel_regularizer=kernel_regularizer)(activation)

def _shortcut(input, residual):
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
    stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))
    equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]
    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Conv2D(filters=residual_shape[CHANNEL_AXIS],
                          kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          padding="same",
                          kernel_initializer="he_normal",
                          kernel_regularizer=l2(0.0001))(input)

    return add([shortcut, residual])


def _residual_block(type,input,filters, repetitions, is_first_layer=False):
	if type == 50 or type == 101 or type == 152:
		blocktype = bottleneck
	if type == 34:
		blocktype = basic_block
	for i in range(repetitions):
		init_strides = (1, 1)
		if i == 0 and not is_first_layer:
		    init_strides = (2, 2)
		input = blocktype(input,filters=filters, init_strides=init_strides,is_first_block_of_first_layer=(is_first_layer and i == 0))
	return input

def bottleneck(input,filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
    """
    Return a final conv layer of filters * 4
    """
    if is_first_block_of_first_layer:
        # don't repeat bn->relu since we just did bn->relu->maxpool
        conv_1_1 = Conv2D(filters=filters, kernel_size=(1, 1),
                          strides=init_strides,
                          padding="same",
                          kernel_initializer="he_normal",
                          kernel_regularizer=l2(1e-4))(input)
    else:
        conv_1_1 = _bn_relu_conv(input,filters=filters, kernel_size=(1, 1),
                                 strides=init_strides)

    conv_3_3 = _bn_relu_conv(conv_1_1,filters=filters, kernel_size=(3, 3))
    residual = _bn_relu_conv(conv_3_3,filters=filters * 4, kernel_size=(1, 1))
    return _shortcut(input, residual)

def basic_block(input,filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
    if is_first_block_of_first_layer:
        # don't repeat bn->relu since we just did bn->relu->maxpool
        conv1 = Conv2D(filters=filters, kernel_size=(3, 3),
                       strides=init_strides,
                       padding="same",
                       kernel_initializer="he_normal",
                       kernel_regularizer=l2(1e-4))(input)
    else:
        conv1 = _bn_relu_conv(input,filters=filters, kernel_size=(3, 3),
                              strides=init_strides)

    residual = _bn_relu_conv(input,filters=filters, kernel_size=(3, 3))
    return _shortcut(input, residual)

###Xnet funcs
def _residual_blockX(type,input,filters, repetitions, is_first_layer=False):
  if type == 50 or type == 101 or type == 152:
    blocktype = bottleneckX
  if type == 34:
    blocktype = basic_blockX
  for i in range(repetitions):
    init_strides = (1, 1)
    if i == 0 and not is_first_layer:
        init_strides = (2, 2)
    input = blocktype(input,filters=filters, init_strides=init_strides,is_first_block_of_first_layer=(is_first_layer and i == 0))
  return input

def bottleneckX(input,filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
    """
    Return a final conv layer of filters * 4
    """
    C = 32
    for i in range(C):
        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv_1_1 = Conv2D(filters=filters*2//C, kernel_size=(1, 1),strides=init_strides,padding="same",kernel_initializer="he_normal",kernel_regularizer=l2(1e-4))(input)
        else:
            conv_1_1 = _bn_relu_conv(input,filters=filters*2//C, kernel_size=(1, 1),
                                     strides=init_strides)
        step_conv_3_3 = _bn_relu_conv(conv_1_1,filters=filters*2//C, kernel_size=(3, 3))
        if i == 0:
        	conv_3_3 = step_conv_3_3
        else:
        	conv_3_3 = concatenate([conv_3_3,step_conv_3_3],axis = 3)
    residual = _bn_relu_conv(conv_3_3,filters=filters * 4, kernel_size=(1, 1))
    return _shortcut(input, residual)

def basic_blockX(input,filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
    C = 32
    for i in range(C):
        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv1 = Conv2D(filters=filters*2//C, kernel_size=(3, 3),
                           strides=init_strides,
                           padding="same",
                           kernel_initializer="he_normal",
                           kernel_regularizer=l2(1e-4))(input)
        else:
            conv1 = _bn_relu_conv(input,filters=filters*2//C, kernel_size=(3, 3),
                                  strides=init_strides)
        step_residual = _bn_relu_conv(conv1,filters=filters, kernel_size=(3, 3))
        if i == 0:
            residual = step_residual
        else:
            residual = add([residual,step_residual])
    return _shortcut(input, residual)


##Resnet
def Resnet(type,input_shape, num_outputs, repetitions):
    input = Input(shape=input_shape)
    conv1 = Conv2D(filters=64,kernel_size = (7,7),strides=(2,2))(input)
    conv1 = _bn_relu(conv1)
    #print(conv1.shape)
    pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(conv1)
    block = pool1
    #print('first',block.shape)
    filters = 64
    for i, r in enumerate(repetitions):
        block = _residual_block(type,block,filters=filters, repetitions=r, is_first_layer=(i == 0))
        #print(block.shape)
        filters *= 2

    # Last activation
    block = _bn_relu(block)

    # Classifier block
    block_shape = K.int_shape(block)
    pool2 = AveragePooling2D(pool_size=(block_shape[ROW_AXIS], block_shape[COL_AXIS]),
                             strides=(1, 1))(block)
    flatten1 = Flatten()(pool2)
    dense = Dense(units=num_outputs, kernel_initializer="he_normal",
                  activation="softmax")(flatten1)

    return input,dense
##ResneXt
def ResneXt(type,input_shape, num_outputs, repetitions):
    input = Input(shape=input_shape)
    conv1 = Conv2D(filters=64,kernel_size = (7,7),strides=(2,2))(input)
    conv1 = _bn_relu(conv1)
    #print(conv1.shape)
    pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(conv1)
    block = pool1
    #print('first',block.shape)
    filters = 64
    for i, r in enumerate(repetitions):
        block = _residual_blockX(type,block,filters=filters, repetitions=r, is_first_layer=(i == 0))
        #print(block.shape)
        filters *= 2

    # Last activation
    block = _bn_relu(block)

    # Classifier block
    block_shape = K.int_shape(block)
    pool2 = AveragePooling2D(pool_size=(block_shape[ROW_AXIS], block_shape[COL_AXIS]),
                             strides=(1, 1))(block)
    flatten1 = Flatten()(pool2)
    dense = Dense(units=num_outputs, kernel_initializer="he_normal",
                  activation="softmax")(flatten1)

    return input,dense


##main
print('startmodeling')
input,dense = ResneXt(type,input_shape=(32, 32, 18), num_outputs = nb_classes, repetitions = repetitions)
model = Model(inputs = input,outputs = dense)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer = 'adam',
              metrics=['accuracy'])

'''
##main
print('startmodeling')
input,dense = Resnet(type,input_shape=(32, 32, 18), num_outputs = nb_classes, repetitions = repetitions)
model = Model(inputs = input,outputs = dense)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer = 'adam',
              metrics=['accuracy'])
'''
def generator(features, labels, batch_size):
 # Create empty arrays to contain batch of features and labels#
    batch_features = np.zeros((batch_size, features.shape[1], features.shape[2], features.shape[3]))
    batch_labels = np.zeros((batch_size,1))
    #labels = labels.reshape(labels.shape[0],labels.shape[3]) #just for cifa-10
    labels = labels.argmax(axis=1)
    while True:
        for i in range(batch_size):
            # choose random index in features
            index= np.random.choice(len(features),1)
            batch_features[i] = features[index]
            batch_labels[i] = labels[index]
        yield batch_features, batch_labels


'''

Xtrain = []
Ttrain = []
for batch in [1,2,3,4,5]:
    D = scipy.io.loadmat('cifar-10-batches-mat/data_batch_%d'%batch)
    Xtrain += [(D['data']/127.5-1.0).reshape([-1,3,32,32]).transpose([0,2,3,1])]
    Ttrain += [(D['labels'][:,numpy.newaxis] == numpy.arange(10)).reshape([-1,1,1,10])*1.0]
    
Xtrain = numpy.concatenate(Xtrain,axis=0)
Ttrain = numpy.concatenate(Ttrain,axis=0)

D = scipy.io.loadmat('cifar-10-batches-mat/test_batch')
Xtest = (D['data'][:500]/127.5-1.0).reshape([-1,3,32,32]).transpose([0,2,3,1])
Ttest = (D['labels'][:500][:,numpy.newaxis] == numpy.arange(10)).reshape([-1,1,1,10])

print(Xtrain.shape,Ttrain.shape)
'''
print('data_loading')
h5f_train = h5py.File('training.h5','r')
Xtrain = np.concatenate((h5f_train['sen1'][:],h5f_train['sen2'][:]),axis = 3)
Ttrain = h5f_train['label'][:]

h5f_test = h5py.File('validation.h5', 'r')
Xtest = np.concatenate((h5f_test['sen1'][:], h5f_test['sen2'][:]), axis=3)
Ttest = h5f_test['label'][:]
'''
Xtest = Xtrain[Xtrain.shape[0]-6863:]
Ttest = Ttrain[Ttrain.shape[0]-6863:]
Xtrain_new = Xtrain[:Xtrain.shape[0]-6863]
Ttrain_new = Ttrain[:Xtrain.shape[0]-6863]
'''
print('training')
model.fit_generator(generator(Xtrain, Ttrain, batch_size),
                    steps_per_epoch = Ttrain.shape[0] // batch_size,
                    validation_data= generator(Xtest, Ttest,batch_size),
                    validation_steps = Ttest.shape[0] // batch_size,
                    epochs=nb_epoch, verbose=1, max_q_size=100,
                    callbacks=[lr_reducer, early_stopper, csv_logger])
##
model.save('final_resneXt50.h5')
