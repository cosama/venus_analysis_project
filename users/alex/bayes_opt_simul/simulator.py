#!bin/env python3
"""
get data and search for best GPR
https://scikit-optimize.github.io/stable/modules/generated/skopt.learning.GaussianProcessRegressor.html
    kernel
        search over and lib will optimize hyperparams for you

    alpha
        search over - dimension sized list

    number of optimizer restarts
        100, just make sure we get the best kernel hyperparams

get best GPR and then search for the best bayesian optimizer on that
    easiest way is to implement our own here because we want to search on time too
    number of available calls
    number of initial randomly sampled points
    acquisition function
        kappa / xi

"""

import itertools

import numpy as np
import skopt
import sklearn # TODO need this for train test split
import scipy # TODO need this for vectorizable gaussian cdf and pdf

import venus_data_utils.database.dbreader as dbreader


# UCB, kappa
# PI, y_opt, xi
# EI, y_opt, xi
class AcquisitionFunction():
    pass


class UpperConfidenceBound(AcquisitionFunction):
    def __init__(self, kappa):
        self._kappa = kappa

    def __str__(self):
        return(f"UCB(kappa={self._kappa})")

    def __call__(self, X, model):
        mean, std = model.predict(X, return_std=True)
        return(mu + self._kappa * std)


class ProbabilityOfImprovement(AcquisitionFunction):
    def __init__(self, xi):
        self._xi = xi

    def __str__(self):
        return(f"PI(xi={self._xi})")

    def __call__(self, X, model, curr_max):
        mean, std = model.predict(X, return_std=True)
        out = scipy.norm.cdf((mean - curr_max - self._xi) / std)
        return(out)


class ExpectedImprovement(AcquisitionFunction):
    def __init__(self, xi):
        self._xi = xi

    def __str__(self):
        return(f"EI(xi={self._xi})")

    def __call__(self, X, model, curr_max):
        mean, std = model.predict(X, return_std=True)
        modified_improvement = (mean - curr_max - self._xi)
        mean_part = modified_improvement * scipy.norm.pdf(modified_improvement / std)
        std_part = std * scipy.norm.cdf(modified_improvement / std)
        out = mean_part + std_part
        return(out)


class BayesOpt:
    def __init__(self, point_gen_dimensions):
        self._point_gen_dimensions = 
        self._best_point_score = float("-inf")

    def ask(self, point):
        pass

    def tell(self, point, score):
        if score > self._best_point_score:
            self._best_point = point
            self._best_point_score = score
        pass


def gpr_search(kernels, alphas, train_x, train_y, test_x, test_y):
    """Evaluate a bunch of gaussian process regression hyperparameters and output their scores."""
    make_gpr = lambda kernel, alpha: skopt.learning.GaussianProcessRegressor(
                    kernel=kernel,
                    alpha=alphas,
                    n_restart_optimizer=num_restarts,
                    normalize_y=True)

    for kernel, alpha in itertools.product(kernels, alphas):
        curr_gpr = make_gpr(kernel, alpha)
        curr_gpr.fit(train_x, train_y)
        gpr_r_squared = curr_gpr.score(test_x, test_y)
        if best_r_squared < r_squared:
            best_gpr_kernel = kernel
            best_gpr_alpha = alpha
            best_gpr_r_squared = r_squared

    return(best_gpr_kernel, best_gpr_alpha, best_gpr_r_squared)


def bayesian_optimization_search(allowed_calls, gpr_samplings, random_calls, acquisition_functions, black_box_model, gpr, gen_func):
    """Evaluate a bunch of bayesian optimization hyperparameters and output their scores."""
    evaluated = {}

    for random_call_num, acquisition_function in itertools.product(random_calls, acquisition_functions):
        # TODO random point generating function
        bayesian_optimizer = BayesOpt(gpr, gen_func, acquisition_function, allowed_calls, random_call_num)

        best_point = bayesian_optimizer.ask()
        best_point_score = black_box_model(best_point)
        for call_num in range(allowed_calls - 1):
            point = bayesian_optimizer.ask()
            score = black_box_model(point)
            bayesian_optimizer.tell(point, score)
            if score > best_point_score:
                best_point = point
                best_point_score = score

        param_string = str({"random_call_num": random_calls, "acquisition_function": acquisition_function})
        evaluated[param_string] = best_point_score

    return(evaluated)


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
    # TODO what is in the pickle file
    # TODO can we aggregate the data with the times and run it all at the same time? probably not bc it may mess with the time dim


    # process data
    test_size = 0.3

    x = np.array([xpts, ypts, zpts]).T
    y = np.array(magpts)
    train_x, test_x, train_y, test_y = sklearn.model_selection.train_test_split(x, y, test_size=test_size)
    dimensions = train_x.dim(0) # TODO check


    # get best gpr hyperparameters
    kernels = [] # kernels
    alphas = [] # alphas
    num_restarts = 10

    evaluated_gprs = gpr_search(kernels, alphas, num_restarts, train_x, train_y, test_x, test_y)

    # TODO
    model = # whatever we think the best model is
    model_untrained = # the best model, but untrained

    # get best bo hyperparameters
    allowed_calls = 20
    gpr_samplings = 1e5

    random_calls = list(range(1, allowed_calls))
    acquisition_functions = [
            # UCB, PI, EI
            ]

    evaluated_bayesian_optimizers = bayesian_optimization_search(
            allowed_calls,
            gpr_samplings,
            random_calls,
            acquisition_functions,
            model,
            model_untrained
            )

