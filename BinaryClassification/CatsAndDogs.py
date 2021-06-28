import os
import warnings
warnings.filterwarnings("ignore")   # 忽略警告信息
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

import matplotlib.pyplot as plt
train_dir = '../data/train_set'
validation_dir = '../data/validation_set'
test_dir = '../test_set'
# 训练集
train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')

# 验证集
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

# 测试集
test_cats_dir = os.path.join(test_dir, 'cats')
test_dogs_dir = os.path.join(test_dir, 'dogs')

# model = tf.keras.models.Sequential([
#     tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
#     tf.keras.layers.MaxPooling2D(2, 2),   # 池化操作，为了将隐藏层h和w进行1/2的操作
#
#     tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
#     tf.keras.layers.MaxPooling2D(2, 2),
#
#     tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
#     tf.keras.layers.MaxPooling2D(2, 2),
#
#     # 为全连接层做准备
#     tf.keras.layers.Flatten(),  # 把特征铺平成一维
#
#     tf.keras.layers.Dense(512, activation='relu'),
#     tf.keras.layers.Dropout(rate=0.4),  # Dropout正则化防止过拟合
#
#     # 二分类sigmoid就够了
#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])
#
# # 配置训练器
# model.compile(loss='binary_crossentropy',
#              optimizer=Adam(lr=1e-4),
#              metrics=['acc'])   #  准确率当做评分标准
#
# # 对训练数据进行预处理，数据进行增强
# train_datagen = ImageDataGenerator(rescale=1./255,
#                                    rotation_range=40,
#                                    width_shift_range=0.2,
#                                    height_shift_range=0.2,
#                                    shear_range=0.2,
#                                    zoom_range=0.2,
#                                    horizontal_flip=True,
#                                    fill_mode='nearest')
# # 归一化
# validation_datagen = ImageDataGenerator(rescale=1./255)
# test_datagen = ImageDataGenerator(rescale=1./255)
#
#
# train_generator = train_datagen.flow_from_directory(
#     train_dir,  # 文件夹路径
#     target_size=(64, 64),  # 指定resize的大小
#     batch_size=30,
#
#     class_mode='binary'
# )
# validation_generator = validation_datagen.flow_from_directory(
#     validation_dir,  # 文件夹路径
#     target_size=(64, 64),   # 指定resize的大小
#     batch_size=20,
#
#     class_mode='binary'
# )
#
# history = model.fit(
#     train_generator,
#     steps_per_epoch=200,  # 6000 images = batch_size * steps
#     epochs=30,
#     validation_data=validation_generator,
#     validation_steps=100,  # 2000images = batch_size * steps
#     verbose=2
# )
# model.save('cat_dog.h5')
# accurate = history.history['acc']
# val_accurate = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
#
# # 绘制 loss 和acc 图像
# epochs = range(len(accurate))
#
# plt.plot(epochs, accurate, 'bo', label='Training accuracy')
# plt.plot(epochs, val_accurate, 'b', label='Validation accuracy')
# plt.title("Training and validation accuracy")
# plt.figure()
#
#
# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title("Training and validation loss")
# plt.legend()
# plt.show()

me = image.load_img('../data/test_set/cats/cat.4013.jpg', target_size=(64, 64))
plt.imshow(me)
me = image.img_to_array(me)
me = np.expand_dims(me, axis=0)  # 添加一个纬度
model = load_model('./cat_dog.h5')
result_c = model.predict(me)
print(result_c)
if round(result_c[0][0]) == 1:
    prediction = 'cat'
else:
    prediction = 'dog'
print(prediction)


