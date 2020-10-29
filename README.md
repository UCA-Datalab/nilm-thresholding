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
|_ scripts
   |_ [python scripts]
```

Credit: [Jack Kelly](https://jack-kelly.com/data/)

## Scripts

The folder [scripts](/scripts) contains one executable script for each
model. Both work the same way. For instance, to use the CONV model, run the
following line on the root folder
(make sure to have the enviroment active and the data downloaded):

```
python scripts/conv_scores.py
```

This will train and score the CONV model using the default parameters.
The script stores several outputs in the [outputs folder](/outputs),
including:
- .txt files with the model scores over the test data
- .png files showing samples of the model's prediction.
- .png files with the scores against the classification weight.

A list with all the available parameters and their default values can be
 seen by running:

```
python scripts/conv_scores.py  --help
```

If you want to use your own set of parameters, there are two options.

1. The easiest way is to call the function with the new value.
For example, if we want to reduce the number of training epochs to 100,
and increase the bacth size to 64, we run:

    ```
    python scripts/conv_scores.py  --epochs 100 --batch-size 64
    ```

    The order and number of parameters does not matter as long as those
    parameters exists on the script (use `--help` to check them out)
    and their new values are of the proper type.

2. Another way of choosing your own parameters is to modify them directly
on the script (below the section Parameters, which is properly indicated).
This is only recommended when modifying complex parameters,
like lists or dictionaries.

### Thresholding methods

There are three threshold methods available. Read [our paper](#publications)
to understand how each threshold works.

- 'mp', Middle-Point
- 'vs', Variance-Sensitive
- 'at', Activation Time

## Contact information

Author: Daniel Precioso, PhD student at Universidad de Cádiz
- Email: daniel.precioso@uca.es
- [Github](https://github.com/daniprec)
- [LinkedIn](https://www.linkedin.com/in/daniel-precioso-garcelan/)
- [ResearchGate](https://www.researchgate.net/profile/Daniel_Precioso_Garcelan)

## Publications

NILM as a regression versus classification problem:
the importance of thresholding

