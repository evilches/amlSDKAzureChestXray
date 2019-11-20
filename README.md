
# Introduction

This project uses Microsoft Azure Machine Learning's SDK to conduct training, local evaluation, and deployment on an image classification model for Chest X-ray diseases. This is an expansion upon an original [project](https://github.com/Azure/AzureChestXRay) that used Azure ML Workbench for the same purpose. For information on the dataset, motivation, and data pre-processing, see this [blog post](https://blogs.technet.microsoft.com/machinelearning/2018/03/07/using-microsoft-ai-to-build-a-lung-disease-prediction-model-using-chest-x-ray-images/), which references the Azure Workbench version of the project.

# Instructions

The SDK is included in the Data Science Virtual Machine (DSVM), which was used to create and run the notebook. Alternatively, a manual set-up [guide](https://docs.microsoft.com/en-us/azure/machine-learning/service/quickstart-create-workspace-with-python) is also available.

Work through the notebooks in the following order:
- code/training.ipynb   
- code/evaluate.ipynb
- code/deployment/deployment.ipynb

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
