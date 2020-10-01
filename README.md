# Better NILM

Non-Intrusive Load Monitoring is aimed to predict the state of each domestic
appliance in a household by knowing the aggregated load of that house.

## Module's scripts

The folder [scripts](/scripts) contains executable scripts. All their
parameters can be tuned by editing the code, which is well explained inside
 each of them. In this version, the scripts only work on UKDALE dataset.

Once you run the script, it performs the following tasks:
1. Load the UKDALE dataset and preprocess it. The functions used in this part
are contained in the folder [ukdale](/better_nilm/ukdale).
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

There are three threshold methods available. Read the paper (to be released)
to understand how each threshold works.

- 'mp', Middle-Point
- 'vs', Variance-Sensitive
- 'at', Activation Time

## Author

Daniel Precioso, [link to github](https://github.com/daniprec)
