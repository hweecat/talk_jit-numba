import numpy as np
from PIL import Image

import os
import sys
import time
from retrying import retry

import zipfile

def unzip_folder(file_name):
  # opening Zip using 'with' keyword in read mode
  with zipfile.ZipFile(file_name, 'r') as file:
    # printing all the information of archive file contents using 'printdir' method
    print(file.printdir())

    # extracting the files using 'extracall' method
    print('Extracting all files...')
    file.extractall()
    print('Done!')

def image_resize(filepath):
  '''Resize and reshape image'''
  # sys.stdout.write('{}: running {}\n'.format(os.getpid(),filepath))
  im = Image.open(filepath)
  resized_im = np.array(im.resize((64,64)))
  # sys.stdout.write('{}: done with {}\n'.format(os.getpid(),filepath))
  return resized_im
 
def square(a_list):
  squared_list = []
  '''Resize and reshape image'''
  for x in a_list:
    squared_list.append(np.square(x))
  return squared_list

def main():
  # assigning filename to a variable
  file_name = 'chest-xray-pneumonia.zip'
  
  if os.path.isdir('./chest_xray') != True:
    unzip_folder(file_name)

  DIR = './chest_xray/train/NORMAL/'

  train_normal = [DIR + name for name in os.listdir(DIR)
            if os.path.isfile(os.path.join(DIR, name))]

  DIR = './chest_xray/train/PNEUMONIA/'

  train_pneumonia = [DIR + name for name in os.listdir(DIR)
            if os.path.isfile(os.path.join(DIR, name))]

  time.sleep(1)
  
  # list comprehension

  start_cpu_time = time.time()

  listcomp_img_output = np.array([image_resize(x) for x in train_normal])

  end_cpu_time = time.time()
  total_tpe_time = end_cpu_time - start_cpu_time
  sys.stdout.write('Elapsed (no numba): {} seconds.\n'.format(total_tpe_time))
  
  time.sleep(1)


  a_list = np.array([i for i in range(1,100000)])

  time.sleep(1)

  # list comprehension

  start_cpu_time = time.time()

  listcomp_array_output = square(a_list)

  end_cpu_time = time.time()
  total_tpe_time = end_cpu_time - start_cpu_time
  sys.stdout.write('Elapsed (no numba): {} seconds.\n'.format(total_tpe_time))

if __name__ == '__main__':
  main()