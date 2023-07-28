from glob import glob
from pathlib import Path

from cellpose import models, io
from natsort import natsorted


def run(images_directory, model_path, use_GPU):
    images_path = Path(images_directory)
    model = models.CellposeModel(gpu=use_GPU, pretrained_model=model_path)
    images = io.get_image_files(images_directory, "")
    channels = [0, 0]
    diam_labels = model.diam_labels.copy()
    for filename in images:
        img = io.imread(filename)
        masks, flows, styles = model.eval(img, diameter=diam_labels, channels=channels)
        # save results so you can load in gui
        io.masks_flows_to_seg(img, masks, flows, model.diam_labels, filename, channels)
        # save results as png
        io.save_masks(img, masks, flows, filename, png=False, tif=True)
