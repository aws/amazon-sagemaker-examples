# XGBoost model inference pipeline with NVIDIA Triton Inference Server on Amazon SageMaker

In this example we show an end-to-end GPU-accelerated fraud detection example making use of tree-based models like XGBoost. In the first notebook [Data Preprocessing using RAPIDS and Training XGBoost for Fraud Detection](1_prep_rapids_train_xgb.ipynb) we demonstrate GPU-accelerated tabular data preprocessing using RAPIDS and training of XGBoost model for fraud detection on the GPU in SageMaker. Then in the second notebook [Pre-processing and XGBoost model inference pipeline with NVIDIA Triton Inference Server on Amazon SageMaker](2_triton_xgb_fil_ensemble.ipynb) we walk through the process of deploying data preprocessing and XGBoost model inference pipeline for high throughput, low-latency inference on Triton in SageMaker. 

## Steps to run the notebooks
1. Launch SageMaker **notebook instance** with `g4dn.xlarge` instance. This example can also be run on a SageMaker **studio notebook instace** but the steps that follow will focus on the **notebook instance**.
    - In **Additional Configuration** select `Create a new lifecycle configuration`. Specify `rapids-2106` as the name in Configuration Setting and copy paste the [on_start.sh](on_start.sh) script as the lifecycle configuration start notebook script. This will create the RAPIDS kernel for us to use inside SageMaker notebook. 
        * For those using AWS on Windows machine, because of the incompatibility between Windows and Unix formatted text, especially in end of line characters you will run into this [error](https://stackoverflow.com/questions/63361229/how-do-you-write-lifecycle-configurations-for-sagemaker-on-windows) if you copy paste [on_start.sh](on_start.sh) script. To prevent that use Notepad++ (or other text editor) to change end of line characters (CRLF to LF) in the [on_start.sh](on_start.sh) script.
            1. Click on Search > Replace (or Ctrl + H)
            2. Find what: \r\n.
            3. Replace with: \n.
            4. Search Mode: select Extended.
            5. Replace All. And then copy paste this into the AWS Lifecycle Configuration Start Notebook UI
    - **IMPORTANT:** In Additional Configuration for **Volume Size in GB** specify at least **50 GB**.
    - For git repositories select the option `Clone a public git repository to this notebook instance only` and specify the Git repository URL https://github.com/aws/amazon-sagemaker-examples/tree/main/sagemaker-triton/fil_ensemble

2. Once JupyterLab is ready, launch the [1_prep_rapids_train_xgb.ipynb](1_prep_rapids_train_xgb.ipynb) notebook with `rapids-2106` conda kernel and run through this notebook to do GPU-accelerated data preprocessing and XGBoost training on credit card transactions dataset for fraud detection use-case. **Make sure to use the `rapids-2106` kernel for this notebook.**

3. Launch the [2_triton_xgb_fil_ensemble.ipynb](2_triton_xgb_fil_ensemble.ipynb) notebook using `conda_python3` kernel (we don't use RAPIDS in this notebook). **Make sure to use the `conda_python3` kernel for this notebook.**  Please note that this notebook requires that the [first notebook](1_prep_rapids_train_xgb.ipynb) be run to create the required dependencies. Run through this notebook to learn how to deploy the ensemble data preprocessing + XGBoost model inference pipeline using the Triton's Python and FIL Backends on Triton SageMaker `g4dn.xlarge` endpoint.
