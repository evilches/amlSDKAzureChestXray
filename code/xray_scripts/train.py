
import sys, os

parser = argparse.ArgumentParser()
parser.add_argument('--data-folder', type=str, dest='data_folder', help='data folder')
args = parser.parse_args()
print(args)
print('Data folder is at:', args.data_folder)
base_dir = args.data_folder
print('List all files: ', os.listdir(args.data_folder))

import src.azure_chestxray_utils as azure_chestxray_utils
import src.azure_chestxray_keras_utils as azure_chestxray_keras_utils

import subprocess

import imgaug as ia
from imgaug import augmenters as iaa
ia.seed(1)

import cv2
import keras.backend as K
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, Callback, ModelCheckpoint
import numpy as np
import pickle
from keras_contrib.applications.densenet import DenseNetImageNet121
from keras.layers import Dense
from keras.models import Model
from keras.utils import multi_gpu_model
from tensorflow.python.client import device_lib
import warnings
from keras.utils import Sequence
import tensorflow as tf
import get_data


def get_available_gpus():
    """

    Returns: list of GPUs available in the system

    """
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
#     try:
#         out_str = subprocess.run(["nvidia-smi", "--query-gpu=gpu_name", "--format=csv"], stdout=subprocess.PIPE).stdout
#         out_list = out_str.decode("utf-8").split('\n')
#         out_list = out_list[1:-1]
#         return out_list
#     except Exception as e:
#         print(e)


# multi GPU model checkpoint. copied from https://github.com/keras-team/keras/issues/8463
class MultiGPUCheckpointCallback(Callback):

    def __init__(self, filepath, base_model, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(MultiGPUCheckpointCallback, self).__init__()
        self.base_model = base_model
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.base_model.save_weights(filepath, overwrite=True)
                        else:
                            self.base_model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('Epoch %05d: %s did not improve' %
                                  (epoch + 1, self.monitor))
            else:
                if self.verbose > 0:
                    print('Epoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.base_model.save_weights(filepath, overwrite=True)
                else:
                    self.base_model.save(filepath, overwrite=True)

# generator for train and validation data
# use the Sequence class per issue https://github.com/keras-team/keras/issues/1638
class DataGenSequence(Sequence):
    def __init__(self, labels, image_file_index, current_state, batch_size):
        self.batch_size = batch_size
        print("BATCH SIZE 2", self.batch_size)
        self.labels = labels
        self.img_file_index = image_file_index
        self.current_state = current_state
        self.len = len(self.img_file_index) // self.batch_size
        print("for DataGenSequence", current_state, "total rows are:", len(self.img_file_index), ", len is", self.len)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # print("loading data segmentation", idx)
        # make sure each batch size has the same amount of data
        current_batch = self.img_file_index[idx * self.batch_size: (idx + 1) * self.batch_size]
        X = np.empty((self.batch_size, resized_height, resized_width, num_channel))
        y = np.empty((self.batch_size, num_classes))

        for i, image_name in enumerate(current_batch):
            path = os.path.join(nih_chest_xray_data_dir, image_name)
            # loading data

            img = cv2.resize(cv2.imread(path), (resized_height, resized_width)).astype(np.float32)
            X[i, :, :, :] = img
            y[i, :] = labels[image_name]

            # only do random flipping in training status
        if self.current_state == 'train':
            x_augmented = seq.augment_images(X)
        else:
            x_augmented = X

        return x_augmented, y


# loss function
def unweighted_binary_crossentropy(y_true, y_pred):
    """
    Args:
        y_true: true labels
        y_pred: predicted labels

    Returns: the sum of binary cross entropy loss across all the classes

    """
    return K.sum(K.binary_crossentropy(y_true, y_pred))


def build_model():
    """

    Returns: a model with specified weights

    """
    # define the model, use pre-trained weights for image_net
    base_model = DenseNetImageNet121(input_shape=(224, 224, 3),
                                     weights='imagenet',
                                     include_top=False,
                                     pooling='avg')

    x = base_model.output
    predictions = Dense(14, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model


if __name__ == "__main__":
    


    #call get_data script
#     base_dir = get_data.download_data()
    print("BASE DIR:", base_dir)
    get_data.preprocess_data(base_dir)
    
    # create the file path variables 
    # paths are typically container level dirs mapped to a host dir for data persistence.
    prj_consts = azure_chestxray_utils.chestxray_consts()

    data_base_input_dir=os.path.join(base_dir, os.path.join(*(prj_consts.BASE_INPUT_DIR_list)))
    data_base_output_dir=os.path.join(base_dir, os.path.join(*(prj_consts.BASE_OUTPUT_DIR_list))) 

    # data used for training
    nih_chest_xray_data_dir=os.path.join(data_base_input_dir, 
                                        os.path.join(*(prj_consts.ChestXray_IMAGES_DIR_list)))

    data_partitions_dir=os.path.join(data_base_output_dir, os.path.join(*(prj_consts.DATA_PARTITIONS_DIR_list))) 

    partition_path = os.path.join(data_partitions_dir, 'partition14_unormalized_cleaned.pickle')
    label_path = os.path.join(data_partitions_dir,'labels14_unormalized_cleaned.pickle')

    # global variables

    weights_dir = os.path.join(data_base_output_dir, os.path.join(*(prj_consts.MODEL_WEIGHTS_DIR_list))) 
    os.makedirs(weights_dir, exist_ok = True)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    # get number of available GPUs
    num_gpu = len(get_available_gpus())
#     num_gpu = 1
    # keras multi_gpu_model slices the data to different GPUs. see https://keras.io/utils/#multi_gpu_model for more details.
    batch_size =num_gpu *  48 # 64 seems t obe too much on NC12 
    print("NUM GPU:", num_gpu)

    # make force_restart = False if you continue a previous train session, make it True to start from scratch
    force_restart = False

    initial_lr = 0.001
    resized_height = 224
    resized_width = 224
    # resized_height = prj_consts.CHESTXRAY_MODEL_EXPECTED_IMAGE_HEIGHT
    # resized_width = prj_consts.CHESTXRAY_MODEL_EXPECTED_IMAGE_WIDTH
    num_channel = 3
    num_classes = 14
    epochs = 250

    seq = iaa.Sequential([
    iaa.Fliplr(0.5),  # horizontal flips
    iaa.Affine(rotate=(-15, 15)),  # random rotate image
    iaa.Affine(scale=(0.8, 1.1)),  # randomly scale the image
    ], random_order=True)  # apply augmenters in random order

    if num_gpu > 1:
        print("using", num_gpu, "GPUs")
        # build model
        with tf.device('/cpu:0'):
            model_single_gpu = build_model()
        # model_single_gpu.load_weights(weights_path)

        # convert to multi-gpu model
        model_multi_gpu = multi_gpu_model(model_single_gpu, gpus=num_gpu)
        model_checkpoint = MultiGPUCheckpointCallback(
            os.path.join(weights_dir, 'azure_chest_xray_14_weights_712split_epoch_{epoch:03d}_val_loss_{val_loss:.4f}.hdf5'),
            model_single_gpu, monitor='val_loss', save_weights_only=True)

    else:
        print("using single GPU")
        model_multi_gpu = build_model()
        model_checkpoint = ModelCheckpoint(
            os.path.join(weights_dir, 'azure_chest_xray_14_weights_712split_epoch_{epoch:03d}_val_loss_{val_loss:.4f}.hdf5'),
            monitor='val_loss', save_weights_only=False)


    num_workers = num_gpu*2 #*10

    model_multi_gpu.compile(optimizer=Adam(lr=initial_lr), loss=unweighted_binary_crossentropy)

    reduce_lr_on_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, min_lr=1e-6)

    callbacks = [model_checkpoint, reduce_lr_on_plateau]

    with open(label_path, 'rb') as f:
        labels = pickle.load(f)

    with open(partition_path, 'rb') as f:
        partition = pickle.load(f)
    
    print("BATCH SIZE 1", batch_size)
    model_multi_gpu.fit_generator(generator=DataGenSequence(labels, partition['train'], current_state='train', batch_size = batch_size),
                                epochs=epochs,
                                verbose=1,
                                callbacks=callbacks,
                                workers=num_workers,
                                # max_queue_size=32,
                                # shuffle=False,
                                validation_data=DataGenSequence(labels, partition['valid'], current_state='validation', batch_size = batch_size)
                                # validation_steps=1
                                )





