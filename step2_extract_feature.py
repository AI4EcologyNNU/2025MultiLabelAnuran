import os
import dill
import yaml

import numpy as np

from anuraset import AnuraSet
from torch.utils.data import DataLoader
from tools import get_file_paths_and_labels, make_sure_path_exists, create_data_frame_AnuraSet

with open('config.yaml', 'r') as file:
    cfg = yaml.safe_load(file)

if __name__ == '__main__':
    AUGMENTED_ANURASET_DIR = r'C:\Users\nnu-xj-group-Tom\Desktop\anuraset-Tom\augmented_audio_and_metadata_file'

    TRAIN_ANNOTATIONS_FILE = os.path.join(AUGMENTED_ANURASET_DIR, 'metadata_augmented.csv')
    TEST_ANNOTATIONS_FILE = os.path.join(AUGMENTED_ANURASET_DIR, 'metadata_test.csv')

    NFOLD = cfg['n_folds']

    save_index_folder = 'split_train_and_validation_dataset'
    save_dataloader_folder = f'dataloader_saved_{cfg["batch_size"]}'

    train_file_paths, train_file_labels = get_file_paths_and_labels(TRAIN_ANNOTATIONS_FILE)
    test_file_paths, test_file_labels = get_file_paths_and_labels(TEST_ANNOTATIONS_FILE)

    for fold in range(NFOLD):
        train_idx = np.load(os.path.join(save_index_folder, 'fold_' + str(fold + 1), 'train.npy'))
        val_idx = np.load(os.path.join(save_index_folder, 'fold_' + str(fold + 1), 'val.npy'))

        train_folder = [train_file_paths[i] for i in train_idx]
        val_folder = [train_file_paths[i] for i in val_idx]
        test_folder = test_file_paths

        train_labels = [train_file_labels[i] for i in train_idx]
        val_labels = [train_file_labels[i] for i in val_idx]
        test_labels = test_file_labels

        train_df = create_data_frame_AnuraSet(train_folder, train_labels)
        val_df = create_data_frame_AnuraSet(val_folder, val_labels)
        test_df = create_data_frame_AnuraSet(test_folder, test_labels)

        training_data = AnuraSet(train_df, train=True)
        validation_data = AnuraSet(val_df, train=False)
        test_data = AnuraSet(test_df, train=False)

        training_loader = DataLoader(training_data,
                                     batch_size=cfg['batch_size'],
                                     shuffle=True,
                                     drop_last=True,
                                     pin_memory=True,
                                     num_workers=cfg['num_workers'],
                                     )
        validation_loader = DataLoader(validation_data,
                                       batch_size=cfg['batch_size'],
                                       shuffle=True,
                                       drop_last=True,
                                       pin_memory=True,
                                       num_workers=cfg['num_workers'],
                                       )
        test_loader = DataLoader(test_data,
                                 batch_size=cfg['batch_size'],
                                 shuffle=False,
                                 drop_last=False,
                                 pin_memory=True,
                                 num_workers=cfg['num_workers'],
                                 )

        save_dataloader_path = os.path.join(save_dataloader_folder, f'fold_{fold + 1}')
        make_sure_path_exists(save_dataloader_path)

        with open(os.path.join(save_dataloader_path, 'training_loader.pkl'), 'wb') as file:
            dill.dump(training_loader, file)
        with open(os.path.join(save_dataloader_path, 'validation_loader.pkl'), 'wb') as file:
            dill.dump(validation_loader, file)
        with open(os.path.join(save_dataloader_path, 'test_loader.pkl'), 'wb') as file:
            dill.dump(test_loader, file)
