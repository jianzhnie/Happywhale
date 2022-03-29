import os

import pandas as pd


def train_data_prepare(root_dir, img_dir, file_name):
    df = pd.read_csv(os.path.join(root_dir, file_name))
    df['image_path'] = root_dir + '/' + img_dir + '/' + df['image']
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


def test_data_prepare(root_dir, img_dir, file_name):

    test_df = pd.read_csv(os.path.join(root_dir, file_name))
    test_df['image_path'] = root_dir + '/' + img_dir + '/' + test_df['image']
    test_df['split'] = 'Test'
    return test_df
