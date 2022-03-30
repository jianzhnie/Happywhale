###
 # @Author: jianzhnie
 # @Date: 2022-03-30 16:09:36
 # @LastEditTime: 2022-03-30 16:19:28
 # @LastEditors: jianzhnie
 # @Description:
 #
###

train_dir = '/home/easyits/road-risk-identification/kaggle/dataset2/cropped_train_images/cropped_train_images/'
test_dir = '/home/easyits/road-risk-identification/kaggle/dataset2/cropped_test_images/cropped_test_images'
train_csv_path = '/home/easyits/road-risk-identification/kaggle/dataset2/train2.csv'
test_csv_path = '/home/easyits/road-risk-identification/kaggle/dataset2/test2.csv'
encoder_classes_path = '/home/easyits/road-risk-identification/kaggle/dataset2/encoder_classes.npy'
train_csv_encoded_folded_path = '/home/easyits/road-risk-identification/kaggle/dataset2/train_encoded_folded.csv'
# 五折交叉验证
n_splits = 5
# 权重保存路径
checkpoints_dir = '/home/easyits/road-risk-identification/kaggle/checkpoints'
# 是否debug
debug = False

sample_submission_csv_path = '/home/easyits/road-risk-identification/kaggle/dataset2/sample_submission.csv'
# 最终提交csv文件
submission_csv_path = './submission.csv'

#  ================Train DataFrame=================
train_df = pd.read_csv(train_csv_path)
train_df['image_path'] = train_df['image'].apply(get_image_path,
                                                    dir=train_dir)

# 给类别编码
encoder = LabelEncoder()
train_df['individual_id'] = encoder.fit_transform(
    train_df['individual_id'])
np.save(encoder_classes_path, encoder.classes_)

# 五折交叉验证
skf = StratifiedKFold(n_splits=n_splits)
for fold, (_, val_) in enumerate(
        skf.split(X=train_df, y=train_df.individual_id)):
    train_df.loc[val_, 'kfold'] = fold

train_df.to_csv(train_csv_encoded_folded_path, index=False)
# ================Test DataFrame================
test_df = pd.read_csv(sample_submission_csv_path)
test_df['image_path'] = test_df['image'].apply(get_image_path,
                                                dir=test_dir)
test_df.drop(columns=['predictions'], inplace=True)

test_df['individual_id'] = 0
test_df.to_csv(test_csv_path, index=False)
