python train.py  --baseroot "D:\Data\Python\neural-enhance\data"  --crop_size 256  --mu 0  --sigma 10  --pre_train True  --save_mode 'epoch'  --save_by_epoch 10  --save_by_iter 100000  --multi_gpu False  --epochs 650  --batch_size 8  --lr 0.0001  --b1 0.9  --b2 0.999  --weight_decay 0.0  --iter_decreased 500000  --lr_decreased 0.00001  --num_workers 8  --pad "zero"  --norm "none"  --in_channels 3  --out_channels 3  --start_channels 32  --m_block 2  --init_type "normal"  --init_gain 0.02
