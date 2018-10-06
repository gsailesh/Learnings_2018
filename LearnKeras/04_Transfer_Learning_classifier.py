import os
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing.image import ImageDataGenerator


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

IM_DIMS = (299,299)
EPOCHS = 2
BATCH_SIZE = 32
NUM_FC_NEURONS = 1024
TRAIN_DIR = '/home/saileshg/sailspace/dev/Projects/GitProjects/Self/Learn/LearnKeras/data/redux/train'
VALIDATE_DIR = '/home/saileshg/sailspace/dev/Projects/GitProjects/Self/Learn/LearnKeras/data/redux/validate'

def get_num_files(path):
    if not os.path.exists(path):
        return 0
    return sum([len(files) for root, dirs, files in os.walk(path)])

def get_num_subfolders(path):
    if not os.path.exists(path):
        return 0
    return sum([len(dirs) for root, dirs, files in os.walk(path)])

def augment_data():
    return ImageDataGenerator(preprocessing_function=preprocess_input, 
                                rotation_range=30, width_shift_range=0.2, 
                                height_shift_range=0.2, shear_range=0.2, 
                                zoom_range=0.2, horizontal_flip=True)


training_data_size = get_num_files(TRAIN_DIR)
print("training_data_size: ", training_data_size)
validation_data_size = get_num_files(VALIDATE_DIR)
NUM_CLASSES = get_num_subfolders(TRAIN_DIR)

train_data_generator = augment_data()
validation_data_generator = augment_data()

train_data_generator = train_data_generator.flow_from_directory(directory=TRAIN_DIR, target_size=IM_DIMS, batch_size=BATCH_SIZE, seed=42)
validation_data_generator = validation_data_generator.flow_from_directory(directory=VALIDATE_DIR, target_size=IM_DIMS, batch_size=BATCH_SIZE, seed=42)

InceptionV3_base = InceptionV3(include_top=False, weights='imagenet')

# Layer definition - Function API style
x = InceptionV3_base.output
x = GlobalAveragePooling2D()(x)
x = Dense(units=NUM_FC_NEURONS, activation='relu')(x)
classification_layer = Dense(units=NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=InceptionV3_base.input, outputs=classification_layer)

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
# model.summary()

history_transfer_learning = model.fit_generator(train_data_generator, epochs=EPOCHS, 
                                    steps_per_epoch=training_data_size // BATCH_SIZE, 
                                    validation_data=validation_data_generator, 
                                    validation_steps=validation_data_size // BATCH_SIZE, 
                                    class_weight='auto')

model.save('/home/saileshg/sailspace/dev/Projects/GitProjects/Self/Learn/LearnKeras/export/inceptionv3-transfer-learning-v0.1.model')