"""
Read data from file using the configuration detailed in the config.json
Perform various methods to clean the data
"""

# Author: Thanh Vu <thanh.vu@csiro.au>

import pandas as pd
import os
import json
import logging
import datetime
import sys
from tqdm import tqdm
import numpy as np

SENTENCE_SEPARATOR = "\n"
RECORD_SEPARATOR = "\n\n"
MULTILABEL_SEPARATOR = "|"


def create_folder_if_not_exist(folder_path: str) -> None:
    """
    Create a folder if not exist
    :param folder_path: str
    :return:
    """
    if not os.path.exists(folder_path):
        try:
            os.makedirs(folder_path)
        except:
            pass
    return None


def create_logger(logger_name: str,
                  file_name: str = None,
                  level: int = logging.INFO,
                  with_console: bool = True) -> logging.RootLogger:
    logging_folder = "{}/../../log_files/{}".format(os.path.dirname(os.path.abspath(__file__)), logger_name)
    create_folder_if_not_exist(logging_folder)

    logging_file = "{}/{}.log".format(logging_folder, datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S.%f")
                                      if file_name is None else file_name)
    logger = logging.getLogger(logger_name)
    hdlr = logging.FileHandler(logging_file)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(level)

    if with_console:
        # show log to console
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(formatter)
        console.setLevel(logging.INFO)
        logger.addHandler(console)
    return logger


def read_config(problem_name: str) -> dict:
    """
    Read the configuration from the config.json file
    :param problem_name: str
        The name of the problem
    :return: configuration: dict
        The configuration of the input problem
    """

    with open("configuration/config.json") as f:
        configuration = json.load(f)
    return configuration[problem_name]


def read_data(configuration, file_path) -> list:
    """
    Read data from configuration file
    """
    id_col_name = configuration["id_col_name"]
    text_col_name = configuration["text_col_name"]
    label_col_names = configuration["label_col_names"]

    if file_path.endswith("csv"):
        data = pd.read_csv(file_path)
    else:
        data = pd.read_excel(file_path)

    id_data = data[id_col_name]
    text_data = data[text_col_name]

    hierarchical_label_data = []
    for label_col_name in label_col_names:
        label_data = data[label_col_name].tolist()
        hierarchical_label_data.append(label_data)
    hierarchical_label_data = np.stack(hierarchical_label_data, axis=1).tolist()

    output = []
    for i in tqdm(range(len(hierarchical_label_data)), desc="Reading data"):
        labels = hierarchical_label_data[i]
        for lvl in range(len(labels)):
            labels[lvl] = str(labels[lvl]).split(MULTILABEL_SEPARATOR)
        output.append((text_data[i], labels, id_data[i]))
    return output



