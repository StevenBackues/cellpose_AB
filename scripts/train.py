import gc
from glob import glob
from pathlib import Path

from cellpose import io, models
from natsort import natsorted


def train(input_data_folder, model_name, use_GPU, n_epochs, min_train_masks=5, learning_rate=0.1, weight_decay=0.0001,
          SGD=True, test_dir=None):
    input_data_path = Path(input_data_folder)
    train_dir = str(input_data_path)
    initial_model = "CPx"
    channels = [0, 0]
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
                save_path=train_dir,
                n_epochs=n_epochs,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                nimg_per_epoch=None,
                SGD=SGD,
                min_train_masks=min_train_masks,
                model_name=model_name)