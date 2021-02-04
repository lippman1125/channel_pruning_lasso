# Channel pruning for accelerating very deep neural networks
This repo contains the PyTorch implementation for paper [**channel pruning for accelerating very deep neural networks**](https://arxiv.org/abs/1707.06168). 

# Prune
We just support vgg-series network pruning, you can type command as follow to execute pruning.  
```
python3 main.py --model vgg16_bn \
                --batch_size 100 \
                --calib_batch 50 \
                --calib_dir /data/imagenet1k_calib \
                --valid_dir /data/ILSVRC2012/val \
                --pruner lasso \
                --ckpt ./vgg16_bn-6c64b313.pth
``` 

|type|layer|sparsity|remain channal|  
|:-----|:------ | :----- |:-----|  
|conv1_2|5|0.5|32|  
|conv2_1|9|0.5|64|
|conv2_2|12|0.5|64|
|conv3_1|16|0.5|130|
|conv3_2|19|0.5|128|
|conv3_3|22|0.5|205|
|conv4_1|26|0.2|409|
|conv4_2|29|0.2|415|
|conv4_3|32|0.2|512|

After pruning:  
&ensp;&ensp;Top1 acc=59.728%  
&ensp;&ensp;Parameter: 135.452 M  
&ensp;&ensp;FLOPs: 7466.797M

# Finetune

```
python3 train.py --data_root /data/ILSVRC2012 \
                 --model vgg16_bn_x \
                 --lr 1e-05 \
                 --lr_type fixed \
                 --dataset imagenet \
                 --n_epoch 10 \
                 --batch_size 128 \
                 --ckpt_path vgg16_pruned.pth.tar \
                 --seed 1
```
# Test

```
python3 test.py --model vgg16_bn_x \
                --checkpoint ./vgg16_pruned_finetune.pth.tar \
                --imagenet_path /data/ILSVRC2012/
```
After finetuning:  
&ensp;&ensp;Top1 acc=69.924%, Top5=89.542%

Orig:  
&ensp;&ensp;Top1 acc=73.476%, Top5=91.536% 
# 