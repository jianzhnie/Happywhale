'''
Author: jianzhnie
Date: 2022-03-29 18:16:39
LastEditTime: 2022-03-29 19:09:28
LastEditors: jianzhnie
Description:

'''

import os

import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder


def get_data(root_dir, image_dir, df_path, n_fold=3):
    df = pd.read_csv(f'{root_dir}/{df_path}')

    df['image_path'] = root_dir + '/' + image_dir + '/' + df['image']

    encoder = LabelEncoder()
    df['individual_id'] = encoder.fit_transform(df['individual_id'])

    skf = StratifiedKFold(n_splits=n_fold)

    for fold, (_, val_) in enumerate(skf.split(X=df, y=df.individual_id)):
        df.loc[val_, 'kfold'] = fold
    return df


if __name__ == '__main__':
    root_dir = '/media/robin/DATA/datatsets/image_data/happywhale'
    img_dir = 'train_images/'
    df_path = 'train.csv'

    df = get_data(root_dir, img_dir, df_path)
    output_file = os.path.join(root_dir, 'datasplit.csv')
    df.to_csv(output_file, index=False)
