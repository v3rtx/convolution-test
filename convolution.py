import os

import tensorflow as tf
import keras_preprocessing
from PIL import ImageFilter
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator, np

model = tf.keras.models.load_model("rps.h5")


# from tkinter.filedialog import askopenfilename
# filename = askopenfilename()

# for fn in uploaded.keys():
# predicting images
# path = filename
# img = image.load_img(path, target_size=(150, 150))
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)

# images = np.vstack([x])
# result = model.predict(images, batch_size=10)
# print(filename)
# print(result)

def classes_to_answer(classes):
    if classes[0][0] == 1:
        return 'paper'
    elif classes[0][1] == 1:
        return 'rock'
    else:
        return 'scissors'


my_dirs = ['./data/my-test-edge']
for my_dir in my_dirs:
    for path in os.listdir(my_dir):
        img = tf.keras.preprocessing.image.load_img(
            os.path.join(my_dir, path),
            color_mode='rgb',
            target_size=(150, 150),
        )
        print(path)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        classes = model.predict(images, batch_size=1)
        print(classes_to_answer(classes))
