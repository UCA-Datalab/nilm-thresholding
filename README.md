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
       conda env create -f env.yml --force
       conda activate better-nilm
       ```
 
### Download UK-DALE

## Module's scripts

The folder [scripts](/scripts) contains executable scripts. All their
parameters can be tuned by editing the code, which is well explained inside
 each of them. In this version, the scripts only work on UK-DALE dataset.

Once you run the script, it performs the following tasks:
1. Load the UK-DALE dataset and preprocess it. The functions used in
this part are contained in the folder [ukdale](/better_nilm/ukdale).
2. Initialise the model and train it, using the functions in
[model](/better_nilm/model).
3. Test the model. Store its scores.
4. Repeat steps 2 and 3 the number of times specified.
5. Create an outputs folder.
6. Average the scores. Store them in the outputs folder.
7. Plot some sample sequences and predictions, using the functions in
[plot_utils](/better_nilm/plot_utils.py)
Store them in the outputs folder.

### Thresholding methods

There are three threshold methods available. Read [our paper](#publications)
to understand how each threshold works.

- 'mp', Middle-Point
- 'vs', Variance-Sensitive
- 'at', Activation Time

## Author

Daniel Precioso, [link to github](https://github.com/daniprec)

## Publications

NILM as a regression versus classification problem:
the importance of thresholding

