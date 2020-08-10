import os
import argparse
import matplotlib
from model_densenet_4channel_fusion import create_model
import numpy as np
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt
from data_lidar_img_fusion_kitti_impainted import DataLoader
import cv2, timeit, csv
from skimage.transform import resize
cmap = plt.cm.jet

# Argument Parser
parser = argparse.ArgumentParser(description='Multi-modal depth estimation')
parser.add_argument('--model',  type=str, help='Trained model file.')
parser.add_argument('--output_path',  type=str, help='output_file_path')
parser.add_argument('--skip',  type=str, help='images to be displayed')


args = parser.parse_args()

# Custom object needed for inference and training

skip = args.skip
output_directory = args.output_path
fieldnames = ['a1', 'a2', 'a3', 'abs_rel', 'sq_rel', 'rmse', 'inference_time']


def create_data_loader():
    dataloader = DataLoader('/scratch/top_view/outputs/carla_files')
    test_dataset = dataloader.get_batched_data_train(
        '/home/sadique/code/train_test_unreal/carla_fog_night_train.txt', args.bs, 8)
    length = dataloader.get_length(
        '/home/sadique/code/train_test_unreal/carla_fog_night_train.txt')

    print('Data loader ready. Number of test images',length)
    return test_dataset


def DepthNorm(x, maxDepth):
    return maxDepth / x

def evaluate(pred, target):

    thresh = np.maximum((target / pred), (pred / target))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    abs_rel = np.mean(np.abs(target - pred) / target)

    sq_rel = np.mean(((target - pred) ** 2) / target)

    rmse = (target - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    error_list = dict(a1=a1, a2=a2, a3=a3, abs_rel=abs_rel, sq_rel=sq_rel, rmse=rmse)

    return error_list


def depth_colorize(depth):
    depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
    depth = 255 * cmap(depth)[:, :, :3]  # H, W, C
    print('depth_colorize', depth)
    return depth.astype('uint8')


def merge_row(input_data, target, pred):
    def preprocess_depth(x):
        y = np.squeeze(x)
        return depth_colorize(y)

    # if is gray, transforms to rgb
    img_list = []

    rgb = np.squeeze(input_data)
    rgb = resize(rgb, (192, 624), anti_aliasing=True)
    #depth_input = rgb[:, :, 3:]
    rgb = 255 * rgb[:, :, :3]
    img_list.append(rgb)
    img_list.append(preprocess_depth(pred))
    img_list.append(preprocess_depth(target))
    img_merge = np.hstack(img_list)
    return img_merge.astype('uint8')


def add_row(img_merge, row):
    return np.vstack([img_merge, row])


def save_image(img_merge, filename):
    image_to_write = cv2.cvtColor(img_merge, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, image_to_write)



def test():
    model = create_model()   # Load model
    model.load_weights(args.model)
    test_dataset = create_data_loader()
    for i, (inputs, y) in enumerate(test_dataset):
        target = DepthNorm(y, 10.0)
        start_1 = timeit.default_timer()
        x = model.predict(inputs)
        pred = np.clip(DepthNorm(x, 10.0), 1, 10.0)
        final_time = timeit.default_timer() - start_1
        target = (target.numpy())

        if i == 0:
            img_merge = merge_row(inputs.numpy(), pred, target)
        elif i % skip == 0 and i < 10 * skip:
            row = merge_row(inputs.numpy(), pred, target)
            img_merge = add_row(img_merge, row)
        elif i == 10 * skip:
            filename = output_directory + 'carla_fog_night_day_test_RGB_addition_huber_14-7_20epochs' + str(i) + '.png'
            save_image(img_merge, filename)

        error_map = evaluate(pred, target)

        with open(output_directory + 'carla_fog_night_day_test_RGB_addition_huber_14-7_20epochs.csv', 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(
                {'a1': error_map['a1'], 'a2': error_map['a2'], 'a3': error_map['a3'], 'abs_rel': error_map['abs_rel'],
                 'sq_rel': error_map['sq_rel'], 'rmse': error_map['rmse'], 'inference_time': final_time})



if __name__ == '__main__':
    test()
