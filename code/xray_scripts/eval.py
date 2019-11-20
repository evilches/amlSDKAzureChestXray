
import sys, os

import src.azure_chestxray_utils as azure_chestxray_utils
import src.azure_chestxray_keras_utils as azure_chestxray_keras_utils


parser = argparse.ArgumentParser()
parser.add_argument('--data-folder', type=str, dest='data_folder', help='data folder')
args = parser.parse_args()
print(args)
print('Data folder is at:', args.data_folder)
base_dir = args.data_folder

import imgaug as ia
from imgaug import augmenters as iaa
ia.seed(1)

import cv2
import keras.backend as K
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, Callback, ModelCheckpoint
import numpy as np
import pandas as pd
import pickle
from keras_contrib.applications.densenet import DenseNetImageNet121
from keras.layers import Dense
from keras.models import Model
from keras.utils import multi_gpu_model
import keras_contrib
from tensorflow.python.client import device_lib
import warnings
from keras.utils import Sequence
import tensorflow as tf
from sklearn import metrics

def get_available_gpus():
    """
    Returns: number of GPUs available in the system
    """
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


# generator for train and validation data
# use the Sequence class per issue https://github.com/keras-team/keras/issues/1638
class DataGenSequence(Sequence):
    def __init__(self, labels, image_file_index, current_state, batch_size):
        self.batch_size = batch_size
        self.labels = labels
        self.img_file_index = image_file_index
        self.current_state = current_state
        self.len = len(self.img_file_index) // self.batch_size
        print("for DataGenSequence", current_state, "total rows are:", len(self.img_file_index), ", len is", self.len)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        print("loading data segmentation", idx)
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
            # this is different from the training code
            x_augmented = X
        else:
            x_augmented = X
        
        return x_augmented, y

if __name__ == "__main__":
    prj_consts = azure_chestxray_utils.chestxray_consts()

    #Organize directories
    data_base_input_dir=os.path.join(base_dir, 
                                     os.path.join(*(prj_consts.BASE_INPUT_DIR_list)))
    data_base_output_dir=os.path.join(base_dir, 
                                      os.path.join(*(prj_consts.BASE_OUTPUT_DIR_list)))

    weights_dir = os.path.join(data_base_output_dir, os.path.join(*(prj_consts.MODEL_WEIGHTS_DIR_list))) 
    fully_trained_weights_dir = os.path.join(data_base_output_dir, os.path.join(*(prj_consts.FULLY_PRETRAINED_MODEL_DIR_list))) 

    nih_chest_xray_data_dir=os.path.join(data_base_input_dir, 
                                         os.path.join(*(prj_consts.ChestXray_IMAGES_DIR_list)))

    data_partitions_dir=os.path.join(data_base_output_dir, 
                                    os.path.join(*(prj_consts.DATA_PARTITIONS_DIR_list)))  
    label_path = os.path.join(data_partitions_dir,'labels14_unormalized_cleaned.pickle')
    partition_path = os.path.join(data_partitions_dir, 'partition14_unormalized_cleaned.pickle')

#     models_file_name= [os.path.join(weights_dir, 
#                                     'azure_chest_xray_14_weights_712split_epoch_300_val_loss_361.7687.hdf5')] 
    models_file_name= [os.path.join(fully_trained_weights_dir, 
                                   'azure_chest_xray_14_weights_712split_epoch_250_val_loss_179-4776.hdf5')] #EDIT THIS ACCORDINGLY

    num_gpu = get_available_gpus()
    # get number of available GPUs
    print("num of GPUs:", len(get_available_gpus()))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    
    resized_height = 224
    resized_width = 224
    num_channel = 3
    num_classes = 14
    batch_size = 512 

    pathologies_name_list = prj_consts.DISEASE_list
    pathologies_name_list

    stanford_result = [0.8094, 0.9248, 0.8638, 0.7345, 0.8676, 0.7802, 0.7680, 0.8887, 0.7901, 0.8878, 0.9371, 0.8047,
                       0.8062, 0.9164]


    with open(label_path, 'rb') as f:
        labels = pickle.load(f)

    with open(partition_path, 'rb') as f:
        partition = pickle.load(f)

    # load test data
    X_test = np.empty((len(partition['test']), 224, 224, 3), dtype=np.float32)
    y_test = np.empty((len(partition['test']) - len(partition['test']) % batch_size, 14), dtype=np.float32)

    for i, npy in enumerate(partition['test']):
        if (i < len(y_test)):
            # round to batch_size
            y_test[i, :] = labels[npy]

    print("len of result is", len(y_test))
    y_pred_list = np.empty((len(models_file_name), len(partition['test']), 14), dtype=np.float32)

    # individual models
    for index, current_model_file in enumerate(models_file_name):
        print(current_model_file)
    #     model = load_model(current_model_file)
        model = azure_chestxray_keras_utils.build_model(keras_contrib.applications.densenet.DenseNetImageNet121); model.load_weights(current_model_file)

        print('evaluation for model', current_model_file)
        # y_pred = model.predict(X_test)

        y_pred = model.predict_generator(generator=DataGenSequence(labels, partition['test'], current_state='test', batch_size = batch_size),
                                         workers=32, verbose=1, max_queue_size=1)
        print("result shape", y_pred.shape)

        # add one fake row of ones in both test and pred values to avoid:
        # ValueError: Only one class present in y_true. ROC AUC score is not defined in that case.
        y_test = np.insert(y_test, 0, np.ones((y_test.shape[1],)), 0)
        y_pred = np.insert(y_pred, 0, np.ones((y_pred.shape[1],)), 0)

        df = pd.DataFrame(columns=['Disease', 'Our AUC Score', 'Stanford AUC Score'])
        scores_list = []
        for d in range(14):
            auc = metrics.roc_auc_score(y_test[:, d], y_pred[:, d])
            df.loc[d] = [pathologies_name_list[d],
                         auc,
                         stanford_result[d]]
            scores_list.append(auc)
        
        df['Delta'] = df['Stanford AUC Score'] - df['Our AUC Score']
        df.to_csv(current_model_file + ".csv", index=False)
        print(df)
