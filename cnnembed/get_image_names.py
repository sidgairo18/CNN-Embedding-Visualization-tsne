import os
import pdb

f = open('image_names_list2.txt')
f2 = open('image_names_vgg16.txt', 'w')
for line in f:
    line = line.strip()
    s = './images/'+line+'\n'
    f2.write(s)
f.close()
f2.close()

