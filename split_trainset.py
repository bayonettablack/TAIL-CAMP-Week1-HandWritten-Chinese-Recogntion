import os
import random
import shutil
from shutil import copy2

root = '/mnt/datasets'
data = 'train'
path = os.listdir(root + '/' + data)
path.sort()
file_train = open('/home/kesci/work/train.txt', 'w+')
file_val = open('/home/kesci/work/val.txt', 'w')

# the percent of train and val
percent = 0.9

# create train.txt val.txt trainval.txt label.txt
for line in path:
	count = 0
	str = root + '/' + data + '/' + line
	num = len(os.listdir(str))
	for child in os.listdir(str):
		str1 = root + '/' + data + '/' + line + '/' + child
		if  count < num * percent:
			file_train.write(str1 + '\n')
		else:
			file_val.write(str1 + '\n')
		count += 1
file_train.close()
file_val.close()
file_train = open('/home/kesci/work/train.txt','r')
file_val = open('/home/kesci/work/val.txt','r')
print(len(file_train.readlines()) + len(file_val.readlines()))
file_train.close()
file_val.close()

'''
# create floder
%%bash
for file in $(ls /mnt/datasets/train/)
do
    echo $file
    mkdir /home/kesci/work/week1-keras/train/$file
done

%%bash
for file in $(ls /mnt/datasets/train/)
do
    echo $file
    mkdir /home/kesci/work/week1-keras/val/$file
done
'''

# copy img to train and val
file_train = open('/home/kesci/work/train.txt','r')
file_val = open('/home/kesci/work/val.txt','r')
for line in file_train.readlines():
    dir_class = line.split('/')[-2]
    dir_path = os.path.join('/home/kesci/work/week1-keras/train', dir_class)
    copy2(line.strip(), dir_path)
file_train.close()
print('train set ok')
for line in file_val.readlines():
    dir_class = line.split('/')[-2]
    dir_path = os.path.join('/home/kesci/work/week1-keras/val', dir_class)
    copy2(line.strip(), dir_path)
file_val.close()
print('val set ok')