import os
import time


def create_dir_for_new_model(name, train_model_path):
    model_name = name + '-' + time.strftime("%Y%m%d-%H%M%S")
    model_dir = os.path.join(train_model_path, model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    return model_dir, model_name


def create_random_model_name(name, checkpoint_path=None):
    if checkpoint_path is None:
        model_name = name + '-rand-' + time.strftime("%Y%m%d-%H%M%S")
    else:
        folder_name = checkpoint_path.split('/')[-2]
        if folder_name.startswith(name):
            model_name = folder_name
        else:
            model_name = name + '-rand-' + time.strftime("%Y%m%d-%H%M%S")
    return model_name
