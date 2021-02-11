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
       conda remove --name better-nilm --all
       conda env create -f environment.yml --force
       conda activate better-nilm
       ```
 
### Download UK-DALE

UK-DALE dataset is hosted on the following link:
[https://data.ukedc.rl.ac.uk/browse/edc/efficiency/residential
/EnergyConsumption/Domestic/UK-DALE-2017/UK-DALE-FULL-disaggregated](https://data.ukedc.rl.ac.uk/browse/edc/efficiency/residential/EnergyConsumption/Domestic/UK-DALE-2017/UK-DALE-FULL-disaggregated)

The files needed by this module are *ukdale.zip* and *ukdale.h5.zip*.
Download both and unzip them in a folder named [data](/data) inside the root.
Once you are done, your local directory should look like this:

```
better_nilm
|_ better_nilm
   |_ [python scripts and subfolders]
|_ data
   |_ ukdale
      |_ [house_1 to house_5]
   |_ ukdale.h5
```

Credit: [Jack Kelly](https://jack-kelly.com/data/)

## Scripts

The folder [better_nilm](/better_nilm) contains an executable script to train the
 models a plot their results. Run the following line on the root folder
(make sure to have the enviroment active and the data downloaded):

```
python better_nilm/train_test_model.py
```

This will train and score the CONV model using the default parameters.
The script stores several outputs in the [outputs folder](/outputs),
including:
- .txt files with the model scores over the test data
- .png files showing samples of the model's prediction.
- .png files with the scores against the classification weight.

The list with all the available parameters and their default values is stored in the
 [configuration file](better_nilm/config.toml).

If you want to use your own set of parameters, duplicate the aforementioned
 configuration file and modify the paremeters you want to change (without deleting any
  parameter). You can then use that config file with the following command:
 
 ```
python better_nilm/train_test_model.py  --path_config <path to your config file>
```

For more information about the script, run:

 ```
python better_nilm/train_test_model.py  --help
```

#### Reproduce paper

To reproduce the results shown in [our paper](#publications), activate the
 environment and then run:

```
nohup sh train_test_sequential.sh > log.out & 
```

This will first create a folder named configs, where the different configurations of the
models are stored. Then, the script `train_test_models.py` will be called, using each
 configuration each. The outputs will be stored in another created folder named outputs.

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
