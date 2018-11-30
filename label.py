import os
# class_names_to_ids = {'daisy': 0, 'dandelion': 1, 'rose': 2,'sunflower':3,'tulip':4}
# data_dir = 'flowers/'
# output_path = 'list.txt'
# fd = open(output_path, 'w')
# for class_name in class_names_to_ids.keys():
#     images_list = os.listdir(data_dir + class_name)
#     for image_name in images_list:
#         fd.write('{}/{} {}\n'.format(class_name, image_name, class_names_to_ids[class_name]))
# fd.close()

import random
_NUM_VALIDATION = 350
_RANDOM_SEED = 0
list_path = 'list.txt'
train_list_path = 'list_train.txt'
val_list_path = 'list_val.txt'
fd = open(list_path)
lines = fd.readlines()
fd.close()
random.seed(_RANDOM_SEED)
random.shuffle(lines)
fd = open(train_list_path, 'w')
for line in lines[_NUM_VALIDATION:]:
    fd.write(line)
fd.close()
fd = open(val_list_path, 'w')
for line in lines[:_NUM_VALIDATION]:
    fd.write(line)
fd.close()