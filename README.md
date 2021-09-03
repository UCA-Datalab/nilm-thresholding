<!-- README template: https://github.com/othneildrew/Best-README-Template -->

<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/UCA-Datalab/nilm-thresholding">
    <img src="images/logo.png" alt="Logo" width="400" height="80">
  </a>

  <h3 align="center">NILM: classification VS regression</h3>
</p>


<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#create-the-environment">Create the Environment</a></li>
      </ul>
    </li>
    <li>
      <a href="#datasets">Datasets</a>
      <ul>
        <li><a href="#uk-dale">UK-DALE</a></li>
      </ul>
      <ul>
        <li><a href="#pecan-street-dataport">Pecan Street Dataport</a></li>
      </ul>
   <li><a href="#preprocess-the-data">Preprocess the Data</a></li>
    </li>
    <li>
      <a href="#train">Train</a>
      <ul>
         <li><a href="#reproduce-the-paper">Reproduce the Paper</a></li>
         <li><a href="#thresholding-methods">Thresholding Methods</a></li>
      </ul>
    </li>
    <li><a href="#publications">Publications</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>

## About the project

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

## Getting started
### Create the Environment

To create the environment using Conda:

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
 
## Datasets

### UK-DALE

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

### Pecan Street Dataport

We are aiming to include this dataset in a future release. You can check the issue here: [https://github.com/UCA-Datalab/nilm-thresholding/issues/8](https://github.com/UCA-Datalab/nilm-thresholding/issues/8)

Any help and suggestions are welcome!

Credit: [Pecan Street](https://dataport.pecanstreet.org/)

## Preprocess the Data

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

### Reproduce the Paper

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

### Thresholding Methods

There are three threshold methods available. Read [our paper](#publications)
to understand how each threshold works.

- 'mp', Middle-Point
- 'vs', Variance-Sensitive
- 'at', Activation Time

## Publications

* [NILM as a regression versus classification problem:
the importance of thresholding](https://www.researchgate.net/project/Non-Intrusive-Load-Monitoring-6)

## Contact

Daniel Precioso - [daniprec](https://github.com/daniprec) -  daniel.precioso@uca.es

Project link: [https://github.com/UCA-Datalab/nilm-thresholding](https://github.com/UCA-Datalab/nilm-thresholding)

ResearhGate link: [https://www.researchgate.net/project/NILM-classification-VS-regression](https://www.researchgate.net/project/NILM-classification-VS-regression)

## Acknowledgements

* [UCA DataLab](http://datalab.uca.es/)
* [David Gómez-Ullate](https://www.linkedin.com/in/david-g%C3%B3mez-ullate-oteiza-87a820b/?originalSubdomain=en)


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/UCA-Datalab/nilm-thresholding.svg?style=for-the-badge
[contributors-url]: https://github.com/UCA-Datalab/nilm-thresholding/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/UCA-Datalab/nilm-thresholding.svg?style=for-the-badge
[forks-url]: https://github.com/UCA-Datalab/nilm-thresholding/network/members
[stars-shield]: https://img.shields.io/github/stars/UCA-Datalab/nilm-thresholding.svg?style=for-the-badge
[stars-url]: https://github.com/UCA-Datalab/nilm-thresholding/stargazers
[issues-shield]: https://img.shields.io/github/issues/UCA-Datalab/nilm-thresholding.svg?style=for-the-badge
[issues-url]: https://github.com/UCA-Datalab/nilm-thresholding/issues
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/daniel-precioso-garcelan/