import os
from loguru import logger


def handle_metainfofile(info_path, root_path='/data2/wzm/N-Caltech/'):
    if not os.path.exists(info_path):
        logger.info(f'File {info_path} does not exist')
        raise FileNotFoundError
    with open(info_path) as f:
        data_list = [sample_path.split(' ') for sample_path in f.readlines()]
        converted_data_list = []
        for data_path, label_path in data_list:
            # if not data_path.startswith(root_path) or not label_path.startswith(root_path):
                logger.info(
                    f'handle the inconsistency between the root path and the data path, {data_path} or {label_path}')
                relative_path = data_path.split(root_path)[-1] + ' ' + label_path.split(root_path)[-1]
                converted_data_list.append(relative_path)

    with open(info_path, 'w') as f:
        f.writelines(converted_data_list)

handle_metainfofile('/data2/wzm/N-Caltech/train.txt')
handle_metainfofile('/data2/wzm/N-Caltech/val.txt')