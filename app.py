import math
from collections import Counter
from Utils import *
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import *
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
import tensorflow.keras
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


# Variable Definition
img_w = 512
img_h = 512
channels = 3
classes = 1
info = 5
grid_w = 16
grid_h = 16


def save_model(model):
    model_json = model.to_json()
    with open("model/text_detect_model.json", "w") as json_file:
        json_file.write(model_json)


def load_model(strr):
    json_file = open(strr, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json, custom_objects={
                                   'leaky_relu': tf.nn.leaky_relu})
    return loaded_model


def yolo_model(input_shape):

    inp = Input(input_shape)

    model = MobileNetV2(
        input_tensor=inp, include_top=False, weights='imagenet')
    last_layer = model.output

    conv = Conv2D(512, (3, 3), activation=tf.nn.leaky_relu,
                  padding='same')(last_layer)
    conv = Dropout(0.4)(conv)
    bn = BatchNormalization()(conv)
    lr = LeakyReLU(alpha=0.1)(bn)

    conv = Conv2D(128, (3, 3), activation=tf.nn.leaky_relu, padding='same')(lr)
    conv = Dropout(0.4)(conv)
    bn = BatchNormalization()(conv)
    lr = LeakyReLU(alpha=0.1)(bn)

    conv = Conv2D(5, (3, 3), activation=tf.nn.leaky_relu, padding='same')(lr)

    final = Reshape((grid_h, grid_w, classes, info))(conv)

    model = Model(inp, final)

    return model


def yolo_loss_func(y_true, y_pred):
    # y_true : 16,16,1,5
    # y_pred : 16,16,1,5
    l_coords = 5.0
    l_noob = 0.5
    coords = y_true[:, :, :, :, 0] * l_coords
    noobs = (-1*(y_true[:, :, :, :, 0] - 1)*l_noob)
    p_pred = y_pred[:, :, :, :, 0]
    p_true = y_true[:, :, :, :, 0]
    x_true = y_true[:, :, :, :, 1]
    x_pred = y_pred[:, :, :, :, 1]
    yy_true = y_true[:, :, :, :, 2]
    yy_pred = y_pred[:, :, :, :, 2]
    w_true = y_true[:, :, :, :, 3]
    w_pred = y_pred[:, :, :, :, 3]
    h_true = y_true[:, :, :, :, 4]
    h_pred = y_pred[:, :, :, :, 4]

    p_loss_absent = K.sum(K.square(p_pred - p_true)*noobs)
    p_loss_present = K.sum(K.square(p_pred - p_true))
    x_loss = K.sum(K.square(x_pred - x_true)*coords)
    yy_loss = K.sum(K.square(yy_pred - yy_true)*coords)
    xy_loss = x_loss + yy_loss
    w_loss = K.sum(K.square(K.sqrt(w_pred) - K.sqrt(w_true))*coords)
    h_loss = K.sum(K.square(K.sqrt(h_pred) - K.sqrt(h_true))*coords)
    wh_loss = w_loss + h_loss

    loss = p_loss_absent + p_loss_present + xy_loss + wh_loss

    return loss


def predict_func(model, inp, iou, name):

    ans = model.predict(inp)

    # np.save('Results/ans.npy',ans)
    boxes = decode(ans[0], img_w, img_h, iou)

    img = ((inp + 1)/2)
    img = img[0]
    # plt.imshow(img)
    # plt.show()

    for i in boxes:

        i = [int(x) for x in i]

        img = cv2.rectangle(
            img, (i[0], i[1]), (i[2], i[3]), color=(0, 255, 0), thickness=2)

    plt.imshow(img)
    plt.show()

    cv2.imwrite(os.path.join('Results', str(name) + '.jpg'), img*255.0)


def train():
    # X and Y numpy arrays are created using the Prepocess.py file

    X = np.load('data/X.npy')
    Y = np.load('data/Y.npy')

    print(X.shape, Y.shape)

    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, train_size=0.75, shuffle=True)
    X = []
    Y = []
    input_size = (img_h, img_w, channels)
    # optimizer

    # checkpoint
    checkpoint = ModelCheckpoint('model/text_detect.h5', monitor='val_loss',
                                 verbose=1, save_best_only=True, mode='min', period=1,
                                 save_freq='epoch')

    model = yolo_model(input_size)
    print(model.summary())

    opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    # model.load_weights('model/text_detect.h5')
    model.compile(loss=yolo_loss_func, optimizer=opt, metrics=['accuracy'])
    hist = model.fit(X_train, Y_train, epochs=10, batch_size=4, callbacks=[
        checkpoint], validation_data=(X_val, Y_val))

    save_model(model)


def main():
    # tf.debugging.set_log_device_placement(True)
    
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    # with tf.device('/GPU:0'):
    #     # Create some tensors
    #     a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    #     b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    #     c = tf.matmul(a, b)

    #     print(c)
    train()

    model = load_model('model/text_detect_model.json')
    model.load_weights('model/text_detect.h5')

    for i in os.listdir('test'):
        img = cv2.imread(os.path.join('test', i))
        img = cv2.resize(img, (512, 512))
        img = (img - 127.5)/127.5
        predict_func(model, np.expand_dims(img, axis=0), 0.3, 'sample')


if __name__ == '__main__':
    main()
