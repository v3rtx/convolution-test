import tensorflow as tf
import os

from PIL import ImageFilter

my_dir = './data/my-test-edge'
# for inner in os.listdir(my_dir):
    # with_inner = os.path.join(my_dir, inner)
for img_path in os.listdir(my_dir):
    full_path = os.path.join(my_dir, img_path)
    img = tf.keras.preprocessing.image.load_img(
        full_path,
        color_mode='rgb',
    )
    img = img.filter(ImageFilter.FIND_EDGES)
    tf.keras.preprocessing.image.save_img(full_path, img)
