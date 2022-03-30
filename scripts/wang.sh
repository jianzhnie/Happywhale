###
 # @Author: jianzhnie
 # @Date: 2022-03-30 16:09:36
 # @LastEditTime: 2022-03-30 17:35:17
 # @LastEditors: jianzhnie
 # @Description:
 #
###

python train.py \
--train_csv_file /home/easyits/road-risk-identification/kaggle/dataset2/train2.csv  \
--test_csv_file  /home/easyits/road-risk-identification/kaggle/dataset2/test2.csv \
--checkpoint_dir word_dir \
--model resnet50  \
--image-size 256 \
--epochs 10 \
--lr 0.01 \
--batch-size 32 \
--pretrained
