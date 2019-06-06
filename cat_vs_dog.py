import glob
import matplotlib.pyplot as plt

from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# get the count of total number of files in the folder
def get_num_files(path):
    if not os.path.exists(path):
        return 0
    return sum([len(files) for r, d, files in os.walk(path)])


#counts number of files in the folders
def get_num_subfolders(path):
    if not os.path.exists(path):
        return 0
    return sum([len(d) for r, d, files in os.walk(path)])


#keras image generator by rotation or slightly

def create_img_generator():
    return ImageDataGenerator(preprocessing_function=preprocess_input, rotation_range=30, width_shift_range=0.2,
                              height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)


#main code
Image_width, Image_height = 299, 299
Training_Epochs = 10
Batch_size = 32
Number_FC_Neurons = 1024

train_dir = './data/train'
validate_dir = './data/validate'

num_train_sample = get_num_files(train_dir)
num_classes = get_num_subfolders(train_dir)
num_validate_samples = get_num_files(validate_dir)


num_epoch = Training_Epochs
batch_size = Batch_size

train_image_gen = create_img_generator()
test_image_gen = create_img_generator()

#training generator
train_generator = train_image_gen.flow_from_directory(train_dir, target_size=(Image_width, Image_height),
                                                      batch_size=batch_size, seed=42)

#validation generator
validation_generator = train_image_gen.flow_from_directory(validate_dir, target_size=(Image_width, Image_height),
                                                      batch_size=batch_size, seed=42)

#load Inception V3 model with pre trains weights
InceptionV3_base_model = InceptionV3(weights='imagenet', include_top=False)  #include_top = false exclude final fully connected layers
print('------------------Inception V3 base model without last FC loaded-----------')


#define the layers in new classification prediction
x = InceptionV3_base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(Number_FC_Neurons, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

#define trainable model for link input from Inception V3 model to new classification prediction
model = Model(input=InceptionV3_base_model.input, outputs=predictions)

#print model structure
print(model.summary())
'''
#option 1: basic transfer learning
print("\n --------Perform transfer learning-----")

# Freeze all the layers in the inception v3 base model
for layer in InceptionV3_base_model.layers:
    layer.trainable = False


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#fit the transfer learning
history_transfer_learning = model.fit_generator(train_generator, epochs=num_epoch, steps_per_epoch=num_train_sample,
                                                validation_data=validation_generator,
                                                validation_steps=num_validate_samples,
                                                class_weight='auto')

model.save('inceptionV3-transfer-learning.model')
'''

#option 2
print("\nFine tuning to existing model")

#freeze
Layers_to_freeze = 172
for layer in model.layers[:Layers_to_freeze]:
    layer.trainable = False
for layer in model.layers[Layers_to_freeze:]:
    layer.trainable = True


model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

history_fine_tuning = model.fit_generator(train_generator, epochs=num_epoch, steps_per_epoch=num_train_sample,
                                          validation_data=validation_generator,
                                          validation_steps=num_validate_samples,
                                          class_weight='auto')

model.save('inceptionV3-fine-tuning.model')
