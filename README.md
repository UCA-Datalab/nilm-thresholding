# Better NILM

Non-Intrusive Load Monitoring is aimed to predict the state of each domestic
appliance in a household only by knowing the aggregated load of that house.
In this project, we implement LSTM-DNN compatible with nilmtk-h5 data files
to monitor any chosen appliance(s).

## Before we start

This project uses the NILM Toolkit (nilmtk) module to process the datasets.
You can download and install that library
[following this link](https://github.com/nilmtk/nilmtk).

The installation process of nilmtk will ask you to create a new conda
environment, which is the one we will be using in this project.We named it
*nilmtk-env*. Activate the environment executing the following line in your
command prompt:
<br/>`$ conda activate nilmtk-env`

# Module's scripts

Take a look at [test/](tests) scripts to get a good idea about how the
library scripts work. Specially look at
[test_model_train](tests/test_model_train.py) to see a full implementation
of the data load and training phase.
