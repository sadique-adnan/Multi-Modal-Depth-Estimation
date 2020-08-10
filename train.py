import os, argparse, math
from model_densenet_4channel_fusion import create_model
from data_lidar_img_fusion_kitti_impainted import DataLoader
import tensorflow as tf
from datetime import datetime
from loss_huber import depth_loss_function
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler, TensorBoard

# Argument Parser
parser = argparse.ArgumentParser(description='Thesis')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--bs', type=int, default=2, help='Batch size')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')


args = parser.parse_args()


def create_data_loader():
    dataloader = DataLoader('/scratch/top_view/outputs/carla_files')
    dataset, val_dataset = dataloader.get_batched_data_train(
        '/home/sadique/code/train_test_unreal/carla_fog_night_train.txt', args.bs, 8)
    length = dataloader.get_length(
        '/home/sadique/code/train_test_unreal/carla_fog_night_train.txt')
    print('Data loader ready.')
    return dataset, val_dataset, length


def step_decay(epoch):
    initial_lrate = args.lr
    drop = 0.2
    epochs_drop = 8.0
    lrate = initial_lrate * math.pow(drop,
           math.floor((1+epoch)/epochs_drop))
    return lrate


def train():
    model = create_model() # Create the model
    dataset, val_dataset, length = create_data_loader()   # create the dataloader
    optimizer = Adam(lr=args.lr, amsgrad=True)  #optimizer
    print('Compiling  model ......')
    model.compile(loss=depth_loss_function, optimizer=optimizer) # Compile the model
    lrate = LearningRateScheduler(step_decay)

    checkpoint_path = "../carla_models_fog/carla_fog_night_day_RGB_huber_addition_14-7_20epochs/cp.ckpt"

    checkpoint_dir = os.path.dirname(checkpoint_path)
    callbacks = []    # Callbacks

    callbacks.append(lrate)

    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss',
                                           verbose=1, save_best_only=False, save_weights_only=True, mode='min',
                                           period=5))
    logdir = "logs_19-7/carla_fog_night_day_RGB_huber_addition_19-7_20epochs" + datetime.now().strftime("%Y%m%d-%H%M%S")
    callbacks.append(TensorBoard(log_dir=logdir))
    print('Ready for training!\n')
    history = model.fit(dataset, epochs=args.epochs, steps_per_epoch=length // args.bs, shuffle=True, callbacks=callbacks,
                        validation_data=val_dataset)   # Start training




if __name__ == '__main__':
    train()





