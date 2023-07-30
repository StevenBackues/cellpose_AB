import gc
from glob import glob
from pathlib import Path

from cellpose import io, models
from natsort import natsorted


def train(train_dir, initial_model, use_GPU, n_epochs, min_train_masks=1, learning_rate=0.1,
          weight_decay=0.0001, test_dir=None, model_name=None, save_path=None, nimg_per_epoch=None):
    """
    Wrapper for cellpose model.train(), I've pruned this a bit to only relevent params for this paper/experiment.
    :param train_dir: str
                what directory are training images and *.npy in

    :param initial_model: str
                either a cellpose general model, such as "CPX" or custom pre-trained model, such as "models/train_set_0/model_1"/

    :param use_GPU: bool

    :param n_epochs: int (default, 500)
                how many times to go through whole training set during training. From model.train()

    :param min_train_masks: int (default, 1)
                minimum number of masks an image must have to use in training set. From model.train()

    :param learning_rate: float or list/np.ndarray (default, 0.2)
                learning rate for training, if list, must be same length as n_epochs. From model.train()

    :param weight_decay: float (default, 0.00001)
                From model.train()

    :param test_dir: str (default, None)
                What directory are testing images and *.npy in

    :param model_name: str (default, None)
                name of network, otherwise saved with name as params + training start time. From model.train()

    :param save_path: string (default, None)
            where to save trained model. If None, will be placed in models/train_dir/*

    :param nimg_per_epoch: int (optional, default None)
            minimum number of images to train on per epoch,
            with a small training set (< 8 images) it may help to set to 8. From to model.train()

    :return:
    """
    channels = [0, 0]
    if save_path is None:
        save_path = "./models/" + train_dir
    # check params
    run_str = f'python -m cellpose --use_gpu --verbose --train --dir {train_dir} --pretrained_model {initial_model} --chan {channels[0]} --chan2 {channels[1]} --n_epochs {n_epochs} --learning_rate {learning_rate} --weight_decay {weight_decay}'
    if test_dir is not None:
        run_str += f' --test_dir {test_dir}'
    run_str += ' --mask_filter _seg.npy'
    print(run_str)

    # actually start training

    # start logger (to see training across epochs)
    logger = io.logger_setup()

    # DEFINE CELLPOSE MODEL (without size model)

    model = models.CellposeModel(gpu=use_GPU, model_type=initial_model)

    # set channels

    # get files
    output = io.load_train_test_data(train_dir, test_dir, mask_filter='_seg.npy')
    train_data, train_labels, _, test_data, test_labels, _ = output
    model.train(train_data, train_labels,
                test_data=test_data,
                test_labels=test_labels,
                channels=channels,
                save_path=save_path,
                n_epochs=n_epochs,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                nimg_per_epoch=nimg_per_epoch,
                SGD=True,
                min_train_masks=min_train_masks,
                model_name=model_name)
