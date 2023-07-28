# I'm running this using the cellpose v2 docker container
# https://hub.docker.com/layers/biocontainers/cellpose/2.1.1_cv2/images/sha256-cfe36943a49590da85c64bb8006330397193de2732faad06f41260296e35978c?context=explore
# cellpose - 2.1.1_cv2
import gc
from pathlib import Path

# basic parameters taken from cellpose collab page
# https://colab.research.google.com/github/MouseLand/cellpose/blob/main/notebooks/run_cellpose_2.ipynb#scrollTo=ldNwr_zxMVha

from cellpose import core
from scripts.test import test

gc.enable()
def test_multiple(folder):
    path = Path(folder)
    items = path.glob('*')
    for item in items:
        if item.is_file():
            print(str(item))
            test(train_dir, str(item), use_GPU)

# check if you have GPU on, see thread if using pycharm
# https://stackoverflow.com/questions/59652992/pycharm-debugging-using-docker-with-gpus
use_GPU = core.use_gpu()
yn = ['NO', 'YES']
print(f'>>> GPU activated? {yn[use_GPU]}')

#
train_dir = "data/Confidential_images_for_MIPAR_100_3folders_fullsize_bb/train/"
"""
dataset = 99 images
dataset ge 5 masks = 62 images
dataset ge 1 masks = 78 images
"""
# training
# train(train_dir, "overtrained", use_GPU, epochs=1000)
# train(train_dir, "overtrained_500", use_GPU, n_epochs=500)
# train(train_dir, "adam", use_GPU, n_epochs=100,SGD=False) failure
# train(train_dir, "min_train_0", use_GPU, n_epochs=100, min_train_masks=0) failure
# train(train_dir, "min_train_1", use_GPU, n_epochs=100, min_train_masks=1)
# train(train_dir, "weight_decay_0-00001", use_GPU, n_epochs=100, weight_decay=0.0001)
# train(train_dir, "weight_decay_0-0001", use_GPU, n_epochs=100, weight_decay=0.001)
# train(train_dir, "learning_rate_0-3", use_GPU, n_epochs=100, learning_rate=0.3)
# train(train_dir, "learning_rate_0-1", use_GPU, n_epochs=100, learning_rate=0.1)
# train(train_dir, "learning_rate_0-1", use_GPU, n_epochs=100, learning_rate=0.1)
train_dir2 = 'data/train/train_159/'
# train(train_dir2, "min_train_1", use_GPU, n_epochs=100, min_train_masks=1)

train_dir3 = 'data/train/train_214/'
test_dir = 'data/testing/test_34/'
# train(train_dir3, "min_train_1_v2", use_GPU, n_epochs=100, min_train_masks=1)
from cellpose import io, models
# test

print("model trained on 193 images, tested on 193 images")
test(train_dir3, train_dir3 + "/models/min_train_1_v2", use_GPU, 193)
print("model trained on 138 images, tested on 193 images")
test(train_dir3, train_dir2 + "/models/min_train_1", use_GPU, 193)
print("model trained on 78 images, tested on 193 images")
test(train_dir3, train_dir + "/models/min_train_1", use_GPU, 193)
print("model trained on 193 images, tested on 34 unknown images")
test(test_dir, train_dir3 + "/models/min_train_1_v2", use_GPU, 34)
print("model trained on 138 images, tested on 34 unknown images")
test(test_dir, train_dir2 + "/models/min_train_1", use_GPU, 34)
print("model trained on 78 images, tested on 34 unknown images")
test(test_dir, train_dir + "/models/min_train_1", use_GPU, 34)

# label_me = "data/Confidential_images_for_MIPAR_100_3folders_fullsize_bb/train"
# run(label_me, train_dir + "/models/honkler", use_GPU)

# test_multiple('data/Confidential_images_for_MIPAR_100_3folders_fullsize_bb/train/models')