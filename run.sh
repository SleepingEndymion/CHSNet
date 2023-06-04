# train VGG16Trans model on FSC
python train.py --tag fsc-baseline  --scheduler step --step 20 --dcsize 8 --batch-size 16 --lr 3e-5 --val-start 100 --val-epoch 10 --resume /mnt/CHSNet/checkpoint/0530_fsc-baseline/160_ckpt.tar
