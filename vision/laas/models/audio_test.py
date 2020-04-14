from keras.layers import Dense,Dropout,Conv3D,Input,MaxPool3D,Flatten,Activation, concatenate
from keras.regularizers import l2
from keras.models import Model
import models.models as m
from keras.optimizers import SGD,Adam

num_classes = 10
model_r3d18, generator_train_batch, generator_val_batch, generator_test_batch = m.r3d_18(num_classes)
model_c3d, generator_train_batch, generator_val_batch, generator_test_batch = m.c3d(num_classes)
x=model_c3d
y=model_r3d18
combined = concatenate([x.output, y.output])
z = Dense(2, activation="relu")(combined)
z = Dense(1, activation="linear")(z)
model = Model(inputs=[x.input, y.input], outputs=z)

lr = 0.005
opt = Adam(lr=lr, decay=0.9)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.summary()