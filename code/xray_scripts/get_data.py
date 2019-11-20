
import sys
import os
import subprocess
import src.azure_chestxray_utils

import pickle
import random
import re
import tqdm

import cv2
import numpy as np
import pandas as pd
import sklearn.model_selection 


def preprocess_data(base_dir):
    #TODO -- get constants from separate file
    prj_consts = src.azure_chestxray_utils.chestxray_consts()

    data_base_input_dir=os.path.join(base_dir, os.path.join(*(prj_consts.BASE_INPUT_DIR_list)))
    data_base_output_dir=os.path.join(base_dir, os.path.join(*(prj_consts.BASE_OUTPUT_DIR_list))) 
    nih_chest_xray_data_dir=os.path.join(data_base_input_dir, os.path.join(*(prj_consts.ChestXray_IMAGES_DIR_list)))
    
    data_partitions_dir=os.path.join(data_base_output_dir, os.path.join(*(prj_consts.DATA_PARTITIONS_DIR_list))) 

    partition_path = os.path.join(data_partitions_dir, 'partition14_unormalized_cleaned.pickle')
    label_path = os.path.join(data_partitions_dir,'labels14_unormalized_cleaned.pickle')

    # global variables

    weights_dir = os.path.join(data_base_output_dir, os.path.join(*(prj_consts.MODEL_WEIGHTS_DIR_list))) 
    
    #check number of images
#     orig_images_no = subprocess.call("find $nih_chest_xray_data_dir -type f | wc -l")
#     print("orig images number:{} ".format(orig_images_no))
    
    #Train/Validation/Test Data partitioning
    total_patient_number = 30805
    NIH_annotated_file = 'BBox_List_2017.csv' # exclude from train pathology annotated by radiologists 
    manually_selected_bad_images_file = 'blacklist.csv'# exclude what viusally looks like bad images
    
    patient_id_original = [i for i in range(1,total_patient_number + 1)]
    
    other_data_dir=os.path.join(nih_chest_xray_data_dir, os.path.join(*(['..','..'])))
    
    # ignored images list is used later, since this is not a patient ID level issue
    ignored_images_set = set()
    with open(os.path.join(other_data_dir, manually_selected_bad_images_file), 'r') as f:
        for line in f:
            # delete the last char which is \n
            ignored_images_set.add(line[:-1])
            if int(line[:-9]) >= 30805:
                print(line[:-1])

    bbox_df = pd.read_csv(os.path.join(other_data_dir, NIH_annotated_file))
    bbox_patient_index_df = bbox_df['Image Index'].str.slice(3, 8)

    bbox_patient_index_list = []
    for index, item in bbox_patient_index_df.iteritems():
        bbox_patient_index_list.append(int(item))

    patient_id = list(set(patient_id_original) - set(bbox_patient_index_list))
    print("len of original patient id is", len(patient_id_original))
    print("len of cleaned patient id is", len(patient_id))
    print("len of unique patient id with annotated data", 
          len(list(set(bbox_patient_index_list))))
    print("len of patient id with annotated data",bbox_df.shape[0])
    
    random.seed(0)
    random.shuffle(patient_id)

    print("first ten patient ids are", patient_id[:10])

    # training:valid:test=7:1:2
    patient_id_train = patient_id[:int(total_patient_number * 0.7)]
    patient_id_valid = patient_id[int(total_patient_number * 0.7):int(total_patient_number * 0.8)]
    # get the rest of the patient_id as the test set
    patient_id_test = patient_id[int(total_patient_number * 0.8):]
    patient_id_test.extend(bbox_patient_index_list)
    patient_id_test = list(set(patient_id_test))


    print("train:{} valid:{} test:{}".format(len(patient_id_train), len(patient_id_valid), len(patient_id_test)))

    # test_set = test_set+left_out_patient_id
    # print("train:{} valid:{} test:{}".format(len(train_set), len(valid_set), len(test_set)))
    
    pathologies_name_list = prj_consts.DISEASE_list
    NIH_patients_and_labels_file = 'Data_Entry_2017.csv'
    
    labels_df = pd.read_csv(os.path.join(other_data_dir, NIH_patients_and_labels_file))
    
    # # create and save train/test/validation partitions list
    
    def process_images(current_df, patient_ids):
        image_name_index = []
        image_labels = {}
        for individual_patient in tqdm.tqdm(patient_ids):
            for _, row in current_df[current_df['Patient ID'] == individual_patient].iterrows():
                processed_image_name = row['Image Index']
                if processed_image_name in ignored_images_set:
                    pass
                else:
                    image_name_index.append(processed_image_name)
                    image_labels[processed_image_name] = np.zeros(14, dtype=np.uint8)
                    for disease_index, ele in enumerate(pathologies_name_list):
                        if re.search(ele, row['Finding Labels'], re.IGNORECASE):
                            image_labels[processed_image_name][disease_index] = 1
                        else:
                            # redundant code but just to make it more readable
                            image_labels[processed_image_name][disease_index] = 0
                    # print("processed", row['Image Index'])
        return image_name_index, image_labels

    train_data_index, train_labels = process_images(labels_df, patient_id_train)
    valid_data_index, valid_labels = process_images(labels_df, patient_id_valid)
    test_data_index, test_labels = process_images(labels_df, patient_id_test)

    print("train, valid, test image number is:", len(train_data_index), len(valid_data_index), len(test_data_index))

    # save the data
    labels_all = {}
    labels_all.update(train_labels)
    labels_all.update(valid_labels)
    labels_all.update(test_labels)

    partition_dict = {'train': train_data_index, 'test': test_data_index, 'valid': valid_data_index}

    with open(os.path.join(data_partitions_dir,'labels14_unormalized_cleaned.pickle'), 'wb') as f:
        pickle.dump(labels_all, f)

    with open(os.path.join(data_partitions_dir,'partition14_unormalized_cleaned.pickle'), 'wb') as f:
        pickle.dump(partition_dict, f)

    # also save the patient id partitions for pytorch training    
    with open(os.path.join(data_partitions_dir,'train_test_valid_data_partitions.pickle'), 'wb') as f:
        pickle.dump([patient_id_train,patient_id_valid,
                     patient_id_test,
                    list(set(bbox_patient_index_list))], f)    

    
