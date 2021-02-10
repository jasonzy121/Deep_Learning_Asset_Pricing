# Deep Learning Asset Pricing Main Code for Training

## Tested Under: 
* **Tensorflow 1.12.0**
* Python 3.6

## Example Code
### Step 1: Training the SDF network
	$ python3 run.py --config=config/config.json --logdir=output --saveBestFreq=128 --printOnConsole=True --saveLog=True --ignoreEpoch=32

### Step 2: Run the first 8 cells of model_GAN.ipynb to generate SDF

### Step 3: Run create_RF_data.py to generate the data with R * F

### Step 4: Train the beta prediction network
	$ python3 run_RtnFcst_ensembles.py --config config_RF --logdir output_RF --task_id 1 --trial_id 1

### Step 5: Run the remaining cells of model_GAN.ipynb to get EV and XS-R2 pricing results

Datasets can be found at [Google Drive](https://drive.google.com/drive/folders/1TrYzMUA_xLID5-gXOy_as8sH2ahLwz-l?usp=sharing)
