from skimage.transform import resize
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

for f in tqdm(os.listdir("./train2014")):
    image = plt.imread("./train2014/{}".format(f))
    resized = resize(image, (32,32), anti_aliasing=True)
    plt.imsave("./resized_small_train2014/{}".format(f), resized)
