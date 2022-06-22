#!/bin/env python3

import math
import statistics
import time

import bayes_opt
import numpy as np

import venus_data_utils.venusplc as venusplc


class VENUS_Bayesian_Optimization():
    def __init__(self, limits_dict, wait_time, sample_time):
        self._bounds = limits_dict
        self._venus = venusplc.VENUSController()
        self.rng = np.random.RandomState(SEED)
        self._wait_time = wait_time
        self._sample_time = sample_time


    def _setpoint(self, dict):
        self._venus.write(dict)
        # TODO damon's complicated (but faster) set currents


    def objective_function(self, params):
        self._setpoint(params)
        time.sleep(self._wait_time)

        fcv1_i_data = []
        end_time = time.time() + self._sample_time
        while time.time() < end_time:
            fcv1_i_data.append(self._venus.read(["fcv1_i"]))

        # all are sample statistics
        mean = statistics.mean(fcv1_i_data)
        standard_deviation = statistics.stdev(fcv1_i_data)
        relative_standard_deviation = sd / mean
        size = len(fcv1_i_data)

        # TODO explain why all the constants
        BEAM_CURR_STD = 30
        instability_cost = BEAM_CURR_STD * 0.5 * (20 * relative_standard_deviation ) ** 2
        output = mean - instability_cost

        # Possible alternative objective functions
        # Signal to Noise Ratio?
        # Lower Confidence Limit?
        # Upper Condifence Limit? (If we want more noise)

        return(output)


if "__main__" == __name__:
    var_bounds = {"inj_i": (120, 130), "ext_i": (97, 110), "mid_i": (95, 107)}
    wait_time = 60 # seconds
    sample_time = 10 # seconds
    SEED = 42

    # TODO
    # Keithly Picoammeter 6485
    # "keithley picoammeter 6485 manual" -> search "rms noise" (make sure not 6487 ammeter)
    # OK, problem, the standard deviation error depends on the order of magnitude output of the ammeter so we can't
    # really set an alpha (variance) for bayesian optimization because it varies by several orders of magnitude
    # Also we would need to set this variance in terms of the objective_function
    # Ask Alex if confused about this comment
    keithley_picoammeter_6485_relative_standard_error = 0.1
    variance = 0.01 # TODO this is wrong

    venus = VENUS_Bayesian_Optimization(var_bounds, SEED, wait_time, sample_time)

    optimizer = bayes_opt.BayesianOptimization(
            f=venus.objective_function,
            random_state=venus.rng,
            pbounds=var_bounds,
            verbose=1)

    optimizer.maximize(
            init_points=5,
            n_iter=30,
            kappa=4.2,
            alpha=variance)

    print("Best result: {}; f(x) = {}.".format(optimizer.max["params"], optimizer.max["target"]))

