import copy
import gc
import logging
import os
import re
import sys

import librosa
import numpy as np
import torchaudio
from datasets import load_dataset
from scipy.io import wavfile

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("prepare_data")


def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = torchaudio.load(batch["path"])

    save_file = os.path.splitext(batch["path"])[0] + '.npy'
    np.save(save_file, speech_array[0].numpy())

    batch["speech"] = save_file
    batch["sampling_rate"] = sampling_rate
    batch["target_text"] = batch["sentence"]
    del speech_array
    gc.collect()
    return batch


def remove_special_characters(batch):
    batch["sentence"] = re.sub(
        CHARS_TO_IGNORE_REGEX, '', batch["sentence"]).lower() + " "
    return batch


def resample_and_save(batch, folder):
    filebase = os.path.basename(batch["speech"])
    filename = os.path.splitext(filebase)[0] + '.wav'

    batch["speech"] = librosa.resample(
        np.array(np.load(batch["speech"])), 48_000, 16_000)
    batch["sampling_rate"] = 16000

    wavfile.write(folder + "/" + filename,
                  batch["sampling_rate"], batch["speech"])
    return batch


def save_transcriptions(dset_path, save_path):
    dst_file = save_path + "/" + "pt-br.trans.txt"

    with open(dst_file, 'w') as f:
        for i in range(len(dset_path)):
            mp3_file = dset_path[i]['path']
            transf = dset_path[i]['sentence'].lower()

            filebase = os.path.basename(mp3_file)
            dst = os.path.splitext(filebase)[0]

            f.write(dst + ' ' + transf)
            f.write('\n')

        f.close()


COLUMNS_TO_REMOVE = ["accent", "age", "client_id",
                     "down_votes", "gender", "locale", "segment", "up_votes"]
CHARS_TO_IGNORE_REGEX = r'[\,\?\.\!\-\;\:\"\“\%\‘\”\�]'


if __name__ == '__main__':

    # Processing train+validation-set

    dataset = load_dataset("common_voice", "pt", split="train+validation")

    logger.info("Removing unused columns from train+validation set")
    dataset = dataset.remove_columns(COLUMNS_TO_REMOVE)

    logger.info("Removing special characters from train+validation set")
    dataset = dataset.map(remove_special_characters)

    dataset_path = copy.deepcopy(dataset)

    logger.info("Converting speech to array train+validation set")
    dataset = dataset.map(speech_file_to_array_fn,
                          num_proc=os.cpu_count(),
                          remove_columns=dataset.column_names)

    logger.info("Resampling and saving train+validation set")
    os.makedirs("data/waves/pt/br")
    dataset = dataset.map(resample_and_save,
                          fn_kwargs={'folder': "data/waves/pt/br"},
                          num_proc=os.cpu_count())

    logger.info("Saving transcriptions of train+validation set")
    save_transcriptions(dataset_path, save_path="data/waves/pt/br")

    # Processing test-set

    test_dataset = load_dataset("common_voice", "pt", split="test")

    logger.info("Removing unused columns from test-set")
    test_dataset = test_dataset.remove_columns(COLUMNS_TO_REMOVE)

    logger.info("Removing special characters from test-set")
    test_dataset = test_dataset.map(remove_special_characters)

    test_dataset_path = copy.deepcopy(test_dataset)

    logger.info("Converting speech to array test-set")
    test_dataset = test_dataset.map(speech_file_to_array_fn,
                                    num_proc=os.cpu_count(),
                                    remove_columns=test_dataset.column_names)

    logger.info("Resampling and saving test-set")
    os.makedirs("data/waves_test/pt/br")
    test_dataset = test_dataset.map(resample_and_save,
                                    fn_kwargs={
                                        'folder': "data/waves_test/pt/br"},
                                    num_proc=os.cpu_count())

    logger.info("Saving transcriptions of test-set")
    save_transcriptions(test_dataset_path, save_path="data/waves_test/pt/br")
