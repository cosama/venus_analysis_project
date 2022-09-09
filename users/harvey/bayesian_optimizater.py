#!/bin/env python3

# TODO delete solenoid_tuning.py when this is confirmed working

import math
import statistics
import time

import bayes_opt
import numpy as np
from dataclasses import dataclass

import venus_data_utils.venusplc as venusplc
import venus_data_utils.database.dbwriter as dbwriter


@dataclass(frozen=True)
class BayesOptParams:
    init_points: int = 5
    n_iter: int = 30
    kappa: float = 4.2
    alpha: float = 0.01
    seed: int = 42

@dataclass(frozen=True)
class VenusParams:
    var_bounds: dict = {"inj_i": (120, 130), "ext_i": (
        97, 110), "mid_i": (95, 107)}
    wait_time: int = 60  # seconds
    sample_time: int = 10  # seconds
    # TODO @alexkireeff set default database_name and table_name
    database_name: str = ""
    table_name: str = ""

class VENUS_Bayesian_Optimizer:
    def __init__(self, params:VenusParams):
        self._bounds = params.var_bounds
        self._venus = venusplc.VENUSController()
        self._wait_time = params.wait_time
        self._sample_time = params.sample_time
        self._table_name = params.table_name

        def flatten(list_of_lists):
            return([item for sub_list in list_of_lists for item in sub_list])

        # Create table and database
        read_names = self._venus.read_vars()
        column_names = flatten(map(lambda x: [x+"_mean", x+"_sd"], read_names))
        column_name_types = {c: "REAL" for c in column_names}
        self._db = dbwriter.DBWriter(params.database_name)
        self._db.create_table(self._table_name, column_name_types)

    def _setpoint(self, requested_vars):
        self.write(requested_vars)
        SOLENOID_VARS = ["inj_i", "ext_i", "mid_i"]  # TODO @damon add sext
        write_solenoid_vars = dict(
            filter(lambda x: x[0] in SOLENOID_VARS, requested_vars.items()))

        self._venus.meta({"fast_sol_i": write_solenoid_vars})

    def objective_function(self, params):
        # TODO catch errors
        self._setpoint(params)
        time.sleep(self._wait_time)

        # Collect statistics
        tmp_data = {}
        end_time = time.monotonic() + self._sample_time
        while time.monotonic() < end_time:
            for var, value in venus.read(read_names).items():
                if var not in tmp_data.keys():
                    tmp_data[var] = []
                tmp_data[var].append(value)

        # aggregate mean and standard deviation
        stats_data = {}
        for name in read_names:
            stats_data[x+"_mean"] = statistics.mean(tmp_data[x])
            stats_data[x+"_sd"] = statistics.stdev(tmp_data[x])

        self._db.add(self._table_name, stats_data)

        # all are sample statistics
        fcv1_micro_i_data = list(map(lambda x: x * 1e6, tmp_data["fcv1_i"]))

        mean = statistics.mean(fcv1_micro_i_data)
        standard_deviation = statistics.stdev(fcv1_micro_i_data)
        relative_standard_deviation = sd / mean
        size = len(fcv1_i_data)

        # TODO explain why all the constants
        # TODO incorporate hyperparameter that determines how important it is to lower noise
        # TODO want to minimize standard deviation given a minimum current output
        BEAM_CURR_STD = 30
        instability_cost = BEAM_CURR_STD * 0.5 * \
            (20 * relative_standard_deviation) ** 2
        output = mean - instability_cost

        # Possible alternative objective functions
        # Signal to Noise Ratio?
        # Lower Confidence Limit?

        return output

    def optimize(self, opt_params: BayesOptParams):
        optimizer = bayes_opt.BayesianOptimization(
            # TODO this needs to get the actual value that it is currently at
            f=self.objective_function,
            random_state=np.random.RandomState(opt_params.seed),
            pbounds=self._bounds,
            verbose=1,
        )
        optimizer.maximize(init_points=opt_params.init_points,
                           n_iter=opt_params.n_iter, kappa=opt_params.kappa, alpha=opt_params.alpha)
        print(
            "Best result: {}; f(x) = {}.".format(
                optimizer.max["params"], optimizer.max["target"]
            )
        )

if "__main__" == __name__:
    # TODO add gui that can be used via X11 forwarding
    # TODO add paramters to a yaml file. GPR model standarlize.

    venus_params = VenusParams()
    opt_params = BayesOptParams(alpha=0.1) # keithley_picoammeter_6485_relative_standard_error = 0.1

    venus_bayesian_opt = VENUS_Bayesian_Optimizer(venus_params)
    venus_bayesian_opt.optimize(opt_params)
