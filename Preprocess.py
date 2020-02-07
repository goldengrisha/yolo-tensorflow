import numpy as np
import cv2
import os
import tqdm
from scipy.io import loadmat
import matplotlib.pyplot as plt


def get_images_and_bb_paths(gt_path, image_path, image_extension='.jpg'):
    train_image_paths = []
    train_gt_paths = []
    for file in tqdm.tqdm(os.listdir(gt_path)):

        file_name, file_extension = os.path.splitext(file)
        image_name = file_name + image_extension

        if 'classes' in file_name:
            continue

        if 'gt' in file:
            image_name = file_name.replace('gt_', '')
            image_name = image_name + image_extension

        path_img = os.path.join(image_path, image_name)
        train_image_paths.append(path_img)
        train_gt_paths.append(os.path.join(gt_path, file))

    return train_image_paths, train_gt_paths


def preprocess_yolo_images(train_image_paths,
                           train_gt_paths,
                           grid_h=16,
                           grid_w=16,
                           img_w=512,
                           img_h=512,
                           show_bb=False):
    X_final = []
    Y_final = []

    for z in tqdm.tqdm(range(len(train_image_paths))):

        new_file = train_image_paths[z]
        x = cv2.imread(train_image_paths[z])
        original_width = x.shape[1]
        original_height = x.shape[0]
        x_sl = img_w / original_width
        y_sl = img_h / original_height

        img = cv2.resize(x, (img_w, img_h))

        X_final.append(img)

        # plt.imshow(cv2.imread(new_file))
        # plt.show()

        i = " "

        if 'img' in new_file:
            i = ", "

        Y = np.zeros((grid_h, grid_w, 1, 5))

        file = train_gt_paths[z]
        name = open(file, 'r')
        data = name.read()
        data = data.split("\n")
        data = data[:-1]

        for li in data:
            temp_list = []
            file_data = li.split(i)
            strr = file_data[4]
            bb = file_data[:5]

            # it should be resized to NN input size
            x = float(bb[1])
            y = float(bb[2])
            w = float(bb[3])
            h = float(bb[4])

            xmin = x*img_w - (w*img_w / 2)
            xmax = x*img_w + (w*img_w / 2)
            ymin = y*img_h - (h*img_h / 2)
            ymax = y*img_h + (h*img_h / 2)

            # setting the proper cell
            x = x * grid_w
            y = y * grid_h

            if show_bb:
                test = cv2.rectangle(
                    img,
                    (int(xmin), int(ymin)),
                    (int(xmax), int(ymax)),
                    color=(0, 255, 0))

            Y[int(y), int(x), 0, 0] = 1
            Y[int(y), int(x), 0, 1] = x - int(x)
            Y[int(y), int(x), 0, 2] = y - int(y)
            Y[int(y), int(x), 0, 3] = w
            Y[int(y), int(x), 0, 4] = h

        if show_bb:
            plt.imshow(test)
            plt.show()

        Y_final.append(Y)

    return X_final, Y_final


def preprocess_images(train_image_paths,
                      train_gt_paths,
                      grid_h=16,
                      grid_w=16,
                      img_w=512,
                      img_h=512,
                      show_bb=False):
    X_final = []
    Y_final = []

    for z in tqdm.tqdm(range(len(train_image_paths))):

        new_file = train_image_paths[z]
        x = cv2.imread(train_image_paths[z])
        original_width = x.shape[1]
        original_height = x.shape[0]
        x_sl = img_w / original_width
        y_sl = img_h / original_height
        img = cv2.resize(x, (img_w, img_h))

        X_final.append(img)

        # plt.imshow(cv2.imread(new_file))
        # plt.show()

        i = " "

        if 'img' in new_file:
            i = ", "

        Y = np.zeros((grid_h, grid_w, 1, 5))

        file = train_gt_paths[z]
        name = open(file, 'r')
        data = name.read()
        data = data.split("\n")
        data = data[:-1]

        for li in data:
            temp_list = []
            file_data = li.split(i)
            strr = file_data[4]
            bb = file_data[:4]

            # it should be resized to NN input size
            xmin = int(bb[0]) * x_sl
            xmax = int(bb[2]) * x_sl
            ymin = int(bb[1]) * y_sl
            ymax = int(bb[3]) * y_sl

            x = ((xmax + xmin) / 2) / img_w
            y = ((ymax + ymin) / 2) / img_h
            w = (xmax - xmin) / img_w
            h = (ymax - ymin) / img_h

            # setting the proper cell
            x = x * grid_w
            y = y * grid_h

            if show_bb:
                test = cv2.rectangle(
                    img,
                    (int(xmin), int(ymin)),
                    (int(xmax), int(ymax)),
                    color=(0, 255, 0))

            # if()

            Y[int(y), int(x), 0, 0] = 1
            Y[int(y), int(x), 0, 1] = x - int(x)
            Y[int(y), int(x), 0, 2] = y - int(y)
            Y[int(y), int(x), 0, 3] = w
            Y[int(y), int(x), 0, 4] = h

        if show_bb:
            plt.imshow(test)
            plt.show()

        Y_final.append(Y)

    return X_final, Y_final


if __name__ == '__main__':
    image_path = 'data/images/'
    gt_path = 'data/ground_truth/'
    train_image_paths, train_gt_paths = get_images_and_bb_paths(
        gt_path, image_path)

    grid_h = 16
    grid_w = 16
    img_w = 512
    img_h = 512

    X_final, Y_final = preprocess_images(
        train_image_paths, train_gt_paths, show_bb=True)

    X = np.array(X_final)
    X_final = []
    Y = np.array(Y_final)
    Y_final = []

    X = (X - 127.5)/127.5

    np.save('data/X.npy', X)
    np.save('data/Y.npy', Y)
