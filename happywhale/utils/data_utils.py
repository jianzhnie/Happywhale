'''
Author: jianzhnie
Date: 2022-03-29 18:16:39
LastEditTime: 2022-03-30 16:50:25
LastEditors: jianzhnie
Description:

'''

import os

import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder


def get_image_path(id: str, dir: str) -> str:
    return f'{dir / id}'


def get_train_data(root_dir, img_dir, df_path, n_fold=3):
    df = pd.read_csv(f'{root_dir}/{df_path}')

    df['image_path'] = root_dir + '/' + img_dir + '/' + df['image']

    encoder = LabelEncoder()
    df['individual_id'] = encoder.fit_transform(df['individual_id'])

    skf = StratifiedKFold(n_splits=n_fold)

    for fold, (_, val_) in enumerate(skf.split(X=df, y=df.individual_id)):
        df.loc[val_, 'kfold'] = fold
    return df


def train_data_clean(root_dir, img_dir, file_name):
    df = pd.read_csv(os.path.join(root_dir, file_name))
    df['split'] = 'Train'
    # convert beluga, globis to whales
    df.loc[df.species.str.contains('beluga'), 'species'] = 'beluga_whale'
    df.loc[df.species.str.contains('globis'),
           'species'] = 'short_finned_pilot_whale'
    df.loc[df.species.str.contains('pilot_whale'),
           'species'] = 'short_finned_pilot_whale'
    df['class'] = df.species.map(lambda x: 'whale'
                                 if 'whale' in x else 'dolphin')

    # fix duplicate labels
    # https://www.kaggle.com/c/happy-whale-and-dolphin/discussion/304633
    df['species'] = df['species'].str.replace('bottlenose_dolpin',
                                              'bottlenose_dolphin')
    df['species'] = df['species'].str.replace('kiler_whale', 'killer_whale')

    return df


def test_data_clean(root_dir, img_dir, file_name):

    test_df = pd.read_csv(os.path.join(root_dir, file_name))
    test_df['image_path'] = root_dir + '/' + img_dir + '/' + test_df['image']
    test_df['individual_id'] = 0
    test_df.drop(columns=['predictions'], inplace=True)
    test_df['split'] = 'Test'
    return test_df


if __name__ == '__main__':
    root_dir = '/media/robin/DATA/datatsets/image_data/happywhale'
    train_img_dir = 'train_images'
    test_img_dir = 'test_images'
    train_file_name = 'train.csv'
    test_file_name = 'sample_submission.csv'
    clean_train_name = 'train_clean.csv'
    clean_test_name = 'test_clean.csv'
    final_file_name = 'train_encoded_folded.csv'

    df = train_data_clean(root_dir, train_img_dir, train_file_name)
    output_file = os.path.join(root_dir, clean_train_name)
    df.to_csv(output_file, index=False)
    df = get_train_data(root_dir, train_img_dir, clean_train_name, n_fold=3)
    output_file = os.path.join(root_dir, final_file_name)
    df.to_csv(output_file, index=False)

    test_df = test_data_clean(root_dir, test_img_dir, test_file_name)
    output_file = os.path.join(root_dir, clean_test_name)
    test_df.to_csv(output_file, index=False)
