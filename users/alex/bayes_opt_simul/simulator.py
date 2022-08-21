#!bin/env python3
"""
get data 

search for best GPR

search for best bayesian optimization given GPR

"""

import itertools

import numpy as np
import sklearn # TODO need this for train test split
import xopt

import venus_data_utils.database.dbreader as dbreader

def search(iterator, data):
    best = None

    for iteration in iterator:
        curr = iteration(data)
        if None == best:
            best = curr
        elif curr > best:
            best = curr


if "__main__" == __name__:
    #db = dbreader(database_name, table_name, rows_to_fetch_per_iteration)
    #aggregate_data = []
    #for data in db:
    #    aggregate_data.extend(data)
    #all_data = np.array(aggregate_data)
    # TODO format the data to work for GPR

    # get data
    import pickle
    data_file_name = "data3.pkl"
    xpts, ypts, zpts, _, magpts = pickle.load(data_file_name)
    # TODO if you want to get multiple data files, you can run a search on each one and then see how well each set of hyperparams does on all of them
    # NOTE we shouldn't run them all with the same bayesian optimizer because the time dependence will start getting weird

    # process data
    test_size = 0.3

    x = np.array([xpts, ypts, zpts]).T
    y = np.array(magpts)
    train_x, test_x, train_y, test_y = sklearn.model_selection.train_test_split(x, y, test_size=test_size)
    dimensions = train_x.dim(0) # TODO check

    # Harvey's code: https://github.com/yubinhu/VENUS-Project/blob/main/Main.ipynb
    # TODO https://github.com/ChristopherMayes/Xopt/blob/main/docs/examples/bayes_opt/mobo.ipynb
    # TODO https://github.com/ChristopherMayes/Xopt/blob/main/docs/examples/bayes_opt/time_dependent_bo.ipynb

    # outline for the following code:

    # use xopt, it's really nice

    # make a generator for all the hyperparams we want to test for GPR

    # test them over all the data we care about (adding the score for each datum)

    # use the best found hyperparams to create a model for the data

    # use the data model to test a bayesian optimizaiton given the best model


