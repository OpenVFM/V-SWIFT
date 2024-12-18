#!/usr/bin/env bash

echo "pip install gdown"

DIR=checkpoint
[ ! -d ${DIR} ] && mkdir ${DIR}

cd checkpoint

gdown --fuzzy 'https://drive.google.com/file/d/181hLvyrrPW2IOGA46fkxdJk0tNLIgdB2/view'
mv checkpoint.pth ssv2_vitb_pretrain_ep800.pth

gdown --fuzzy 'https://drive.google.com/file/d/1xZCiaPF4w7lYmLt5o1D5tIZyDdLtJAvH/view'
mv checkpoint.pth ssv2_vitb_finetune_ep800.pth

gdown --fuzzy 'https://drive.google.com/file/d/1I18dY_7rSalGL8fPWV82c0-foRUDzJJk/view'
mv checkpoint.pth ssv2_vitb_pretrain_ep2400.pth

gdown --fuzzy 'https://drive.google.com/file/d/1dt_59tBIyzdZd5Ecr22lTtzs_64MOZkT/view'
mv checkpoint.pth ssv2_vitb_finetune_ep2400.pth