import tensorflow as tf
import numpy as np
import random


class DataLoader():
    def __init__(self, data_path):
        self.data_path = data_path
        self.shape_rgb = [672, 1248]
        self.shape_depth = [336, 624]
        self.shape_sparse_radar = [672, 1248]
      
    def get_length(self, filename):
        #self.filename = filenames_file
        with open(filename, 'r') as f:
            filenames = f.readlines()
        length = len(filenames)
        print(length)
        return length*0.9
        
    def parse_function_train(self, line):
        split_line = tf.compat.v1.string_split([line]).values
      
        image_path = tf.strings.join([self.data_path, split_line[0]])
        depth_gt_path = tf.strings.join([self.data_path, tf.strings.strip(split_line[1])])
        image = tf.image.decode_png(tf.io.read_file(image_path), channels = 3)
        radar_path = tf.strings.join([self.data_path, tf.strings.strip(split_line[2])])
      
        sparse_radar = tf.image.decode_png(tf.io.read_file(radar_path), channels=1)
        sparse_radar = tf.cast(sparse_radar, tf.float32) / 256.0
        depth_gt = tf.image.decode_png(tf.io.read_file(depth_gt_path), channels = 1)

        image = tf.image.convert_image_dtype(image, tf.float32)
        depth_gt = tf.image.convert_image_dtype(depth_gt, tf.float32)
        sparse_radar = tf.image.convert_image_dtype(sparse_radar, tf.float32)
              
        image  = tf.image.resize(image, self.shape_rgb, tf.image.ResizeMethod.AREA)
        sparse_radar  = tf.image.resize(sparse_radar, self.shape_sparse_radar)
       
        depth_gt = tf.image.resize(depth_gt , self.shape_depth)
        input_data = tf.concat([image, sparse_radar], axis=2)
        depth_gt = 50 / tf.clip_by_value(depth_gt * 50, 1, 50)
        return input_data, depth_gt
    
    
    
    def parse_function_test(self, line):
        split_line = tf.compat.v1.string_split([line]).values
        image_path = tf.strings.join([self.data_path, split_line[0]])
        depth_gt_path = tf.strings.join([self.gt_path, tf.strings.strip(split_line[1])])
        image = tf.image.decode_png(tf.io.read_file(image_path), channels = 3)
        radar_path = tf.strings.join([self.sparse_radar_path, tf.strings.strip(split_line[2])])
     
        sparse_radar = tf.image.decode_png(tf.io.read_file(radar_path), channels=1, dtype=tf.uint16)
        sparse_radar = tf.cast(sparse_radar, tf.float32) / 256.0
     
        depth_gt = tf.image.decode_png(tf.io.read_file(depth_gt_path), channels = 1)
       
        
        image = tf.image.convert_image_dtype(image, tf.float32)
        depth_gt = tf.image.convert_image_dtype(depth_gt, tf.float32)
        #sparse_radar = tf.image.convert_image_dtype(sparse_radar, tf.float32)
      
      
        print('image',image)
              
        image  = tf.image.resize(image, self.shape_rgb, tf.image.ResizeMethod.AREA)
        sparse_radar  = tf.image.resize(sparse_radar, self.shape_sparse_radar)
       
        depth_gt = tf.image.resize(depth_gt , self.shape_depth)
        
        
        input_data = tf.concat([image, sparse_radar], axis=2)
        
        depth_gt = 50 / tf.clip_by_value(depth_gt * 50, 1, 50)
        
        return input_data, depth_gt


    def get_batched_data_train(self, filename, batch_size=4, num_threads=8):
        with open(filename, 'r') as f:
            filenames = f.readlines()
            print('filenames',filenames)

        
        filenames.sort()
        random.seed(500)
        random.shuffle(filenames)

        split = int(0.9 * len(filenames))
        train_filenames = filenames[:split]
        print(len(train_filenames))
        val_filenames = filenames[split:]
        print(len(val_filenames))
        self.dataset = tf.data.Dataset.from_tensor_slices(filenames)
        self.val_dataset = tf.data.Dataset.from_tensor_slices(val_filenames)
        self.dataset = self.dataset.shuffle(buffer_size=20000)
       
        self.dataset = self.dataset.map(self.parse_function_train, num_parallel_calls=num_threads)
        self.val_dataset = self.val_dataset.map(self.parse_function_train, num_parallel_calls=num_threads)

        
        self.dataset = self.dataset.map(self.train_preprocess, num_parallel_calls=num_threads)
        print('parse',self.dataset)
        self.dataset = self.dataset.batch(batch_size=batch_size)
        self.dataset = self.dataset.repeat()
        self.val_dataset = self.val_dataset.batch(batch_size=batch_size)
        print(self.dataset)

        self.dataset = self.dataset.prefetch(batch_size)
        return self.dataset, self.val_dataset
            
     
    
    def get_batched_data_test(self,  test_filename, batch_size=1, num_threads=1,):
        with open(test_filename, 'r') as f:
            test_filenames = f.readlines()
        
        self.test_dataset = tf.data.Dataset.from_tensor_slices(test_filenames)
        self.test_dataset = self.test_dataset.map(self.parse_function_test, num_parallel_calls=1)
        self.dataset = self.test_dataset.shuffle(buffer_size=20000)

        self.test_dataset = self.test_dataset.batch(batch_size=1)
        self.test_dataset = self.test_dataset.prefetch(1)
        
        return self.test_dataset
        
        
        
    def train_preprocess(self, input_data, depth_gt):
        
        # Random flipping
        do_flip = tf.random.uniform([], 0, 1)
        input_data = tf.cond(do_flip > 0.5, lambda: tf.image.flip_left_right(input_data), lambda: input_data)
        depth_gt = tf.cond(do_flip > 0.5, lambda: tf.image.flip_left_right(depth_gt), lambda: depth_gt)
      
        # Random gamma, brightness, color augmentation
        
        do_augment = tf.random.uniform([], 0, 1)
        input_data = tf.cond(do_augment > 0.5, lambda: self.augment_image(input_data), lambda: input_data)


        return input_data, depth_gt
    
    
    @staticmethod
    def augment_image(input_data):
        # gamma augmentation
        gamma = tf.random.uniform([], 0.9, 1.1)
        image_aug = input_data ** gamma

        # brightness augmentation
        brightness = tf.random.uniform([], 0.75, 1.25)
        image_aug = input_data * brightness


        return image_aug 



