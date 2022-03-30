<!--
 * @Author: jianzhnie
 * @Date: 2022-03-30 08:55:17
 * @LastEditTime: 2022-03-30 17:32:41
 * @LastEditors: jianzhnie
 * @Description:
 *
-->
# Happywhale


### DataPreprocess

```sh
python happywhale/utils/data_utils.py
```

### Train  Model

```sh
python train.py \
--train_csv_file /media/robin/DATA/datatsets/image_data/happywhale/train_encoded_folded.csv \
--test_csv_file  /media/robin/DATA/datatsets/image_data/happywhale/test_clean.csv \
--checkpoint_dir word_dir \
--model resnet50  \
--image-size 256 \
--epochs 10 \
--lr 0.01 \
--batch-size 32 \
--pretrained
```
