import os
import yaml
import numpy as np

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from tools import make_sure_path_exists, get_file_paths_and_labels

with open('config.yaml', 'r') as file:
    cfg = yaml.safe_load(file)

if __name__ == '__main__':
    AUGMENTED_ANURASET_DIR = r'C:\Users\nnu-xj-group-Tom\Desktop\anuraset-Tom\augmented_audio_and_metadata_file'

    TRAIN_ANNOTATIONS_FILE = os.path.join(AUGMENTED_ANURASET_DIR, 'metadata_augmented.csv')

    AUGMENTED_AUDIO_DIR = os.path.join(AUGMENTED_ANURASET_DIR, 'augmented_audio')

    NFOLD = cfg['n_folds']

    augmented_train_file_paths, augmented_train_file_labels = get_file_paths_and_labels(TRAIN_ANNOTATIONS_FILE)

    mskf = MultilabelStratifiedKFold(n_splits=NFOLD, shuffle=True, random_state=42)

    save_index_folder = 'split_train_and_validation_dataset'

    for fold, (train_idx, val_idx) in enumerate(mskf.split(augmented_train_file_paths, augmented_train_file_labels)):
        print(f"Fold {fold + 1}")
        print("Train Index:", train_idx)
        # print(len(train_idx))
        print("Validation Index:", val_idx)

        save_index_path = os.path.join(save_index_folder, f'fold_{fold + 1}')
        make_sure_path_exists(save_index_path)

        print(len(train_idx), len(val_idx))

        np.save(os.path.join(save_index_path, 'train.npy'), train_idx)
        np.save(os.path.join(save_index_path, 'val.npy'), val_idx)
