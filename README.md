# NILM: classification VS regression

Non-Intrusive Load Monitoring (NILM)  aims to predict the status
or consumption of  domestic appliances in a household only by knowing
the aggregated power load. NILM can be formulated as regression problem
or most often as a classification problem. Most datasets gathered
by smart meters allow to  define naturally a regression problem,
but the corresponding classification problem  is a derived one,
since it requires a conversion from the power signal to the status of each
device by a thresholding method. We treat three different thresholding
methods to perform this task, discussing their differences on various
devices from the UK-DALE dataset. We analyze the performance of
deep learning state-of-the-art architectures on both the regression and
classification problems, introducing criteria to select the most convenient
thresholding method.

Source: [see publications](#publications)

## Set up
### Create the environment using Conda

  1. Install miniconda
     
     ```
     curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh | bash
     ```

     Say yes to everything and accept default locations. Refresh bash shell with `bash -l`

  2. Update conda
     
      ```
      conda update -n base -c defaults conda
      ```

  3. Clone this repository and cd into the folder

  4. Create and activate conda environment (removing previously existing env of the same name)
     
       ```
       conda remove --name nilm-thresholding --all
       conda env create -f environment.yml --force
       conda activate nilm-thresholding
       ```
 
## Data

### UK-DALE

#### Download UK-DALE

UK-DALE dataset is hosted on the following link:
[https://data.ukedc.rl.ac.uk/browse/edc/efficiency/residential
/EnergyConsumption/Domestic/UK-DALE-2017/UK-DALE-FULL-disaggregated](https://data.ukedc.rl.ac.uk/browse/edc/efficiency/residential/EnergyConsumption/Domestic/UK-DALE-2017/UK-DALE-FULL-disaggregated)

The files needed by this module are *ukdale.zip* and *ukdale.h5.zip*.
Download both and unzip them in a folder named [data](/data) inside the root.
Once you are done, your local directory should look like this:

```
nilm-thresholding
|_ nilmth
   |_ [python scripts and subfolders]
|_ data
   |_ ukdale
      |_ [house_1 to house_5]
   |_ ukdale.h5
```

Credit: [Jack Kelly](https://jack-kelly.com/data/)

### Preprocess

Once downloaded the raw data from any of the sources above,
you must preprocess it.
This is done by running the following script:

```
python nilmth/preprocess.py
```

This will generate a new folder, named 'data-prep',
containing all the preprocessed data.

## Train

The folder [nilmth](/nilmth) contains an executable script to train the
 models. Run the following line on the root folder
(make sure to have the enviroment active and the data downloaded):

```
python nilmth/train.py
```

This will train the CONV model using the default parameters.
The script stores several outputs in the [outputs folder](/outputs),
including:
- .txt files with the model scores over the validation data
- .pth files containing the model weights
- .png files with a section of the validation data and the model's prediction

The list with all the available parameters and their default values is stored in the
 [configuration file](nilmth/config.toml).

If you want to use your own set of parameters, duplicate the aforementioned
 configuration file and modify the paremeters you want to change (without deleting any
  parameter). You can then use that config file with the following command:
 
 ```
python nilmth/train.py  --path_config <path to your config file>
```

For more information about the script, run:

 ```
python nilmth/train.py  --help
```

Once the models are trained, test them with:

 ```
python nilmth/test.py  --path_config <path to your config file>
```

#### Reproduce paper

To reproduce the results shown in [our paper](#publications), activate the
 environment and then run:

```
nohup sh train_sequential.sh > log.out & 
```

This will first create a folder named configs, where the different configurations of the
models are stored. Then, the script `train.py` will be called, using each
 configuration each. This will store the model weights, which will be used
 again during the test phase:
 
 ```
nohup sh test_sequential.sh > log.out & 
```

### Thresholding methods

There are three threshold methods available. Read [our paper](#publications)
to understand how each threshold works.

- 'mp', Middle-Point
- 'vs', Variance-Sensitive
- 'at', Activation Time

## Publications

[NILM as a regression versus classification problem:
the importance of thresholding](https://www.researchgate.net/project/Non-Intrusive-Load-Monitoring-6)

## Contact information

Author: Daniel Precioso, PhD student at Universidad de Cádiz
- Email: daniel.precioso@uca.es
- [Github](https://github.com/daniprec)
- [LinkedIn](https://www.linkedin.com/in/daniel-precioso-garcelan/)
- [ResearchGate](https://www.researchgate.net/profile/Daniel_Precioso_Garcelan)
