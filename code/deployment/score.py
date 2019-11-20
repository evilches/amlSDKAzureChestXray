
from azureml.core.model import Model
import os, sys, pickle, base64
from keras.layers import Dense
import keras_contrib
from keras_contrib.applications.densenet import DenseNetImageNet121
import numpy as np
import json
import pandas as pd
import azure_chestxray_utils, azure_chestxray_keras_utils, azure_chestxray_cam

#Parameters 
global as_string_b64encoded_pickled_data_column_name
as_string_b64encoded_pickled_data_column_name   = 'encoded_image'

def init():
    global model
    model_path = Model.get_model_path("chest_xray_model_179.hdf5")
    model = azure_chestxray_keras_utils.build_model(keras_contrib.applications.densenet.DenseNetImageNet121)
    model.load_weights(model_path)
    
def run(data, with_cam = False): 
    input_cv2_image = np.array(json.loads(data)['data'])
    predictions, serialized_cam_image, predicted_disease_index, direct_result = get_image_score_and_serialized_cam(input_cv2_image)

    outDict = {"predictedDiseaseIndex": int(predicted_disease_index), "chestXrayScore": str(predictions)}
    
    if with_cam:
        outDict["chestXrayCAM"]= as_string_b64encoded_pickled(serialized_cam_image)
        
    return json.dumps(outDict)
        

####################################
# Utils
####################################
def as_string_b64encoded_pickled(input_object):
     #b64encode returns bytes class, make it string by calling .decode('utf-8')
    return (base64.b64encode(pickle.dumps(input_object))).decode('utf-8')

def unpickled_b64decoded_as_bytes(input_object):
    if input_object.startswith('b\''):
        input_object = input_object[2:-1]
    # make string bytes
    input_object   =  input_object.encode('utf-8')
    #decode and the unpickle the bytes to recover original object
    return (pickle.loads(base64.b64decode(input_object)))

def get_image_score_and_serialized_cam(crt_cv2_image):
    prj_consts = azure_chestxray_utils.chestxray_consts()
    crt_cv2_image = azure_chestxray_utils.normalize_nd_array(crt_cv2_image)
    crt_cv2_image = 255*crt_cv2_image
    direct_result = model.predict(np.expand_dims(crt_cv2_image,0))
    crt_cv2_image=crt_cv2_image.astype('uint8')
    predictions, cam_image, predicted_disease_index = \
    azure_chestxray_cam.get_score_and_cam_picture(crt_cv2_image, model)
    blended_image = azure_chestxray_cam.process_cam_image(cam_image, crt_cv2_image)
    serialized_image = azure_chestxray_cam.plot_cam_results(blended_image, cam_image, crt_cv2_image, \
                 prj_consts.DISEASE_list[predicted_disease_index])
    return predictions, serialized_image, predicted_disease_index, direct_result

#for testing only
def test_image():
    resized_height = 224
    resized_width = 224
    
    #init local model
    global model
    model = azure_chestxray_keras_utils.build_model(keras_contrib.applications.densenet.DenseNetImageNet121)
    model.load_weights('azure_chest_xray_14_weights_712split_epoch_250_val_loss_179-4776.hdf5')
    
    #script for later use
    import cv2
    image_dir = "./../../../data/chestxray/ChestX-ray8/ChestXray-NIHCC/images"
    image_name = "00000003_000.png"
    full_path = os.path.join(image_dir, image_name)
    cv2_image = cv2.resize(cv2.imread(full_path), (resized_height, resized_width))
    
#     ans = model.predict(np.expand_dims(cv2_image,0))
    
    test_images = json.dumps({"data": cv2_image.tolist()})
    test_images = bytes(test_images, encoding = "utf8")
    ans = run(test_images, True)
    
    return ans
