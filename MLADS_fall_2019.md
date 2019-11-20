
# Introduction

This project uses Microsoft Azure Machine Learning's SDK to conduct training, local evaluation, and deployment on an image classification model for Chest X-ray diseases. This is an expansion upon an original [project](https://github.com/Azure/AzureChestXRay) that used Azure ML Workbench for the same purpose. For information on the dataset, motivation, and data pre-processing, see this [blog post](https://blogs.technet.microsoft.com/machinelearning/2018/03/07/using-microsoft-ai-to-build-a-lung-disease-prediction-model-using-chest-x-ray-images/), which references the Azure Workbench version of the project. 

# Instructions  

The SDK is included in the Data Science Virtual Machine (DSVM), which was used to create and run the notebook. Alternatively, a manual set-up [guide](https://docs.microsoft.com/en-us/azure/machine-learning/service/quickstart-create-workspace-with-python) is also available.

Work through the notebooks in the following order:
- code/training.ipynb   
- code/evaluate.ipynb
- code/deployment/deployment.ipynb

### Repositories  
https://github.com/georgeAccnt-GH/AzureChestXRayNoAML/tree/ghiordan/mladsfall19_01  
https://github.com/Azure/amlSDKAzureChestXray  
https://github.com/Azure/AzureChestXRay (Azure Chest Xray v1  - Azure ML Workbench bases)  
  
### Data  
https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community  
https://nihcc.app.box.com/v/ChestXray-NIHCC/  
[batch_download_zips.py](https://nihcc.app.box.com/v/ChestXray-NIHCC/file/371647823217)  
  
Azure Storage container info will be provided (see [000_setup_azure_sa.ipynb](https://github.com/georgeAccnt-GH/AzureChestXRayNoAML/blob/ghiordan/mladsfall19_01/code/000_setup_azure_sa.ipynb)).   
  

### Setup 

#### Provision an Ubuntu DSVM   
Via portal (__NC12_Promo__, you can later resize to __NC24_Promo__) or via az cli script below:    
https://github.com/microsoft/AMLSDKRModelsOperationalization/blob/master/code/amlsdk_operationalization/AzureVMviaAzureCLI.sh  
  
Full DSVM setup instructions here:  
https://github.com/microsoft/AMLSDKRModelsOperationalization#prerequisites  
(Win cli version of the above script, also instructions on how to open ports for ssh and Jupyter Notebook sessions)   
   
#### Docker  
##### Docker install  

Docker evolution: nvidia docker run -> docker -runtime nvidia -> docker run --gpus all    
Goal is to be able to run:  
```
docker run --gpus all nvidia/cuda:9.0-base nvidia-smi
```

Resources:
#https://github.com/NVIDIA/nvidia-docker  
#https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-16-04  

```
docker version
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
#sudo systemctl restart docker
sudo service docker restart
sudo systemctl status docker
docker run --gpus all nvidia/cuda:9.0-base nvidia-smi
```
Tips: 
 - https://www.guguweb.com/2019/02/07/how-to-move-docker-data-directory-to-another-location-on-ubuntu/   
 - loginvm022@ghiordangpu02vm:~$ sudo usermod -a -G docker $USER  
  
##### Docker setup  
https://github.com/georgeAccnt-GH/PowerAIWithDocker/wiki
```
sudo mkdir -p /datadrive01/prj
sudo chmod -R ugo=rwx  /datadrive01/
sudo mkdir -p /data_dir
sudo chmod -R ugo=rwx  /data_dir/
docker login
#sudo adduser <your_login_name> docker
#sudo groupadd docker
# newgrp docker
sudo usermod -a -G docker $USER
```
##### Local env setup  
```
sudo apt-get update
pip install --upgrade pip
sudo apt-get install tmux
pip install -U python-dotenv
```

##### Clone repos  
```
cd /datadrive01/prj/  
git clone https://github.com/georgeAccnt-GH/AzureChestXRayNoAML.git
cd /datadrive01/prj/
git clone https://github.com/Azure/amlSDKAzureChestXray.git
```


# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
