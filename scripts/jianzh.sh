###
 # @Author: jianzhnie
 # @Date: 2022-03-30 16:09:45
 # @LastEditTime: 2022-03-30 17:29:09
 # @LastEditors: jianzhnie
 # @Description:
 #
###

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
