import csv
import random
import shutil
from pathlib import Path

from cellpose import core

from scripts.log_wrapper import IOWrapper
from scripts.train_wrapper import get_masks


def write_list_to_csv(data_list, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data_list)


def create_strata(data_list):
    # Step 1: Identify distinct values for stratification
    distinct_strata = list(set(item[1] for item in data_list))
    # Step 2: Divide data_list into sublists based on stratification
    stratified_data = {stratum: [] for stratum in distinct_strata}
    for item in data_list:
        stratified_data[item[1]].append(item)
    # Step 3: Randomly select samples from each stratum
    stratified_sample = []
    sample_size_per_stratum = 8 // len(distinct_strata)  # Adjust as needed
    for stratum, data_subset in stratified_data.items():
        if len(data_subset) >= sample_size_per_stratum:
            stratified_sample.extend(random.sample(data_subset, sample_size_per_stratum))
        else:
            stratified_sample.extend(data_subset)
    # If necessary, randomly select additional samples to meet the desired sample size
    remaining_samples = 8 - len(stratified_sample)
    stratified_sample.extend(random.sample(data_list, remaining_samples))
    return stratified_sample


def move_samples(stratified_sample, target_dir):
    for path, _ in stratified_sample:
        source_path = Path(path)
        target_path = target_dir / source_path.name
        shutil.move(source_path, target_path)
        source_path = source_path.parent / Path(source_path.stem + "_seg.npy")
        target_path = target_dir / source_path.name
        shutil.move(source_path, target_path)


io_wrapper = IOWrapper()
logger, log_file, log_handler = io_wrapper.logger_setup(log_directory="./data/log/")

# check if you have GPU on, see thread if using pycharm
# https://stackoverflow.com/questions/59652992/pycharm-debugging-using-docker-with-gpus
use_GPU = core.use_gpu()
yn = ['NO', 'YES']
logger.info(f'>>> GPU activated? {yn[use_GPU]}')

# train dirs
train_set_0_2B_dir = Path('data/train/set_0_2B/')  # 78 IMAGES
train_set_0_B_dir = Path('data/train/set_0_B/')  # 78 IMAGES

train_set_1_dir = Path('data/train/set_1/')  # 115 IMAGES
# test dirs
test_set_0_2B_dir = Path('data/test/test_8_set_0_2B/')  # 8 IMAGES
test_set_0_B_dir = Path('data/test/test_8_set_0_B/')  # 8 IMAGES
test_set_1_dir = Path('data/test/test_34_set_1/')  # 34 IMAGES
# empty dirs
empty_set_0_dir = Path('data/test/empty_set_0/')  # 21 IMAGES
empty_set_1_dir = Path('data/test/empty_set_1/')  # 12 IMAGES

list_of_dirs = [train_set_0_2B_dir, train_set_0_B_dir, train_set_1_dir, test_set_0_2B_dir, test_set_0_B_dir,
                test_set_1_dir, empty_set_0_dir, empty_set_1_dir]

for dir in list_of_dirs:
    nmasks = get_masks(dir, True)
    logger.info(nmasks)
    csv_filename = dir.name + ".csv"
    write_list_to_csv(nmasks, "data/mask/" + csv_filename)

# basic sample set creation. ideally you would stratify your samples.
# nmasks = get_masks(train_set_0_B_dir, True)
# random_sel = random.sample(nmasks, 8)
# print(random_sel)
#
# csv_filename = 'train_set_0_B_dir.csv'
# write_list_to_csv(nmasks, csv_filename)
# print(nmasks)


# strata example, could use scikit. DO NOT UNCOMMENT move_samples unless you want to move samples!

# nmasks = get_masks(train_set_0_B_dir, True)
# logger.info(f'nmasks in {train_set_0_B_dir}: {nmasks}')
# stratified_sample = create_strata(nmasks)
# logger.info(f'stratified sample of {train_set_0_B_dir}: {stratified_sample}')
# # move_samples(stratified_sample, test_set_0_B_dir)
#
# nmasks = get_masks(train_set_0_2B_dir, True)
# logger.info(f'nmasks in {train_set_0_2B_dir}: {nmasks}')
# stratified_sample = create_strata(nmasks)
# logger.info(f'stratified sample of {train_set_0_2B_dir}: {stratified_sample}')
# # move_samples(stratified_sample, test_set_0_2B_dir)
