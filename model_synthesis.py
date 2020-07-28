from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
# import cv2      // for resizing the image

if(os.path.exists("./home/final_model.hdf5")):
    model = load_model("./home/final_model.hdf5")  # Best Model
    model._make_predict_function()


def make_simpsons(cnt):

    if(cnt == 1):
        img_count = 5
        noise_dims = 100
        rnd = np.random.randint(100)
        if(os.path.exists("./static/predictedImages")):
            shutil.rmtree("./static/predictedImages")
        plt.figure(figsize=(15, 3))

        for i in range(img_count):
            noise_vector = np.random.normal(0, 1, size=(1, noise_dims))
            img = model.predict(noise_vector)
            img[img < 0] = 0
            plt.subplot(1, img_count, i+1)
            plt.imshow(img[0])
            plt.axis("off")

        os.mkdir("./static/predictedImages")
        plt.savefig("./static/predictedImages/image%d.jpg" % rnd)
        plt.clf()
        return rnd

    else:
        l = os.listdir("./static/predictedImages/")
        num = l[0][5:7]
        if(num[1] == '.'):
            num = num[0]
        num = int(num)
        return num
