#!/bin/env python3

# TODO delete solenoid_tuning.py when this is confirmed working

import math
import statistics
import time

import bayes_opt
import numpy as np

import venus_data_utils.venusplc as venusplc
import venus_data_utils.database.dbwriter as dbwriter


class VENUS_Bayesian_Optimization:
    def __init__(self, limits_dict, wait_time, sample_time, database_name, table_name):
        self._bounds = limits_dict
        self._venus = venusplc.VENUSController()
        self.rng = np.random.RandomState(SEED)
        self._wait_time = wait_time
        self._sample_time = sample_time
        self._database_name = database_name
        self._table_name = table_name

    def _setpoint(self, requested_vars):
        # TODO should use PID for this and this done on the PLC
        # if requested solenoid current is further away than USE_FAST_FUNC Amps, use the fast solenoid setting function
        USE_FAST_FUNC = 0.1  # Amps

        # when requesting a solenoid current, overshoot by 5 Amps to reach the desired current faster
        OVERSHOOT_AMPS = 5  # Amps

        # solenoid variables for which to use the fast solenoid setting function
        SOLENOID_VARS = ["inj_i", "ext_i", "mid_i"]

        # fast func is done when we are within the range specified below of
        # requested_vars for the solenoids (range depends on which side we come from)
        # TODO why do these exist when USE_FAST_FUNC exists
        # it should be one or the other
        diffup = dict(zip(SOLENOID_VARS, [0.03, 0.04, 0.08]))  # Amps
        diffdown = dict(zip(SOLENOID_VARS, [0.06, 0.10, 0.25]))  # Amps

        # Medium Error
        # TODO name this better? or maybe use something else
        MEDIUM_DIFF = 0.08  # Amps

        # Overshoot when there is MEDIUM_DIFF
        MEDIUM_OVERSHOOT = 0.01  # Amps

        # time it takes for to time out after overshooting
        TIMED_OUT_TIME = 5 * 60  # seconds

        # Threshold at which we determine solenoid is done and can be set to the requested value
        # Threshold is for the average difference between the actual current and the requested current over the last
        # ABS_DIFF_LEN samplings
        AVG_DIFF_DONE_THRESH = 0.04  # Amps
        # Value that we initialize the abs_diff (absolute difference) arrays to
        ABS_DIFF_INIT = 5  # Amps
        # Length of the abs_diff (absoulte difference) arrays
        ABS_DIFF_LEN = 40

        # set initial settings
        self._venus.write(requested_vars)

        read_solenoid_i = venus.read(SOLENOID_VARS)

        # to see which direction the requested current is in
        def get_diff(var):
            return (var, requested_vars[var] - read_solenoid_i[var])

        diff = dict(map(get_diff, SOLENOID_VARS))

        # to see for which solenoids we must overshoot
        # because they are currently far away
        def is_done(solenoid_var):
            return (var, abs(diff[solenoid_var]) > USE_FAST_FUNC)

        solenoid_done = dict(map(is_done, SOLENOID_VARS))

        # to see what to overshoot the solenoid current to
        def overshoot(solenoid_var):
            tmp = requested_vars[solenoid_var] + solenoid_done[
                solenoid_var
            ] * math.copysign(OVERSHOOT_AMP, diff[var])
            return (solenoid_var, tmp)

        set_solenoid_i = dict(map(overshoot, SOLENOID_VARS))

        # set solenoids
        # the ones that aren't overshooting will be set to the requested value
        self._venus.write(set_solenoid_i)

        # to determine when done overshooting
        # so when we get within diffup[var] or diffdown[var]
        # (depending on whether we are going up or down)
        def done_current(solenoid_var):
            if diff[var] < 0:
                return (var, requested_vars[solenoid_var] - diffup[solenoid_var])
            else:  # diff[var] > 0
                return (var, requested_vars[solenoid_var] + diffdown[solenoid_var])

        done_solenoid_i = dict(map(done_current, SOLENOID_VARS))

        abs_diff = dict(
            map(
                lambda solenoid_var: (
                    solenoid_var,
                    [
                        ABS_DIFF_INIT,
                    ]
                    * ABS_DIFF_LEN,
                ),
                SOLENOID_VARS,
            )
        )

        def check_done(solenoid_var):
            is_done = (
                diff[solenoid_var]
                * (read_solenoid_i[solenoid_var] - done_solenoid_i[solenoid_var])
                > 0
            )
            if not solenoid_done[solenoid_var] and is_done:
                self._venus.write({solenoid_var: requested_vars[solenoid_var]})
                solenoid_done[solenoid_var] = True

        done = all(
            statistics.mean(abs_diff[solenoid_var]) <= AVG_DIFF_DONE_THRESH
            for solenoid_var in SOLENOID_VARS
        )

        while True:
            start_time = time.monotonic()
            while TIMED_OUT_TIME < time.monotonic() - start_time:
                # TODO why repeat 5 times?
                for _ in range(5):
                    time.sleep(0.1)
                    for solenoid_var, read_value in venus.read(SOLENOID_VARS).items():
                        read_solenoid_var[solenoid_var] = read_value
                        check_done(solenoid_var)

                # TODO slight difference between the solenoid code because solenoid_var is updated after if not done check, but it shouldn't make a difference (keep the comment here until this code is confirmed working)
                for solenoid_var in SOLENOID_VARS:
                    abs_diff[solenoid_var] = abs_diff[solenoid_var][1:]
                    abs_diff[solenoid_var].append(read_solenoid_var[solenoid_var])

                done = all(
                    statistics.mean(abs_diff[solenoid_var]) <= AVG_DIFF_DONE_THRESH
                    for solenoid_var in SOLENOID_VARS
                )
                if done:
                    return

                for solenoid_var in SOLENOID_VARS:
                    # if small error
                    # TODO if small error, shouldn't we just be done?
                    # and if the error isn't small enough, shouldn' t we just call it big error
                    if (
                        not done[solenoid_var]
                        and abs(
                            read_solenoid_var[solenoid_var]
                            - requested_vars[solenoid_var]
                        )
                        < MEDIUM_DIFF
                    ):
                        # TODO here we use current direction instead of initial direction for some reason
                        # instead I think it would be better to use initial direction, like in the rest of the code
                        set_solenoid_i[solenoid_var] = set_solenoid_i[
                            solenoid_var
                        ] - MEDIUM_OVERSHOOT * math.copysign(
                            1,
                            read_solenoid_var[solenoid_var]
                            - requested_vars[solenoid_var],
                        )

    def objective_function(self, params):
        self._setpoint(params)
        time.sleep(self._wait_time)

        # Create table and database
        read_names = venus.read_vars()
        column_names = flatten(map(lambda x: [x+"_mean", x+"_sd"], read_names))
        column_name_types = {c: "REAL" for c in column_names}
        db = dbwriter.DBWriter(self._database_name)
        db.create_table(self._table_name, column_name_types)


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

        db.add(self._table_name, stats_data)

        # all are sample statistics
        fcv1_micro_i_data = list(map(lambda x: x * 1e6, tmp_data["fcv1_i"]))

        mean = statistics.mean(fcv1_micro_i_data)
        standard_deviation = statistics.stdev(fcv1_micro_i_data)
        relative_standard_deviation = sd / mean
        size = len(fcv1_i_data)

        # TODO explain why all the constants
        BEAM_CURR_STD = 30
        instability_cost = BEAM_CURR_STD * 0.5 * (20 * relative_standard_deviation) ** 2
        output = mean - instability_cost

        # Possible alternative objective functions
        # Signal to Noise Ratio?
        # Lower Confidence Limit?

        return output


if "__main__" == __name__:
    var_bounds = {"inj_i": (120, 130), "ext_i": (97, 110), "mid_i": (95, 107)}
    wait_time = 60  # seconds
    sample_time = 10  # seconds
    SEED = 42

    # TODO
    # "keithley Picoammeter 6485 Manual" -> search "rms noise"
    # problem: standard deviation is dependent on the current being measured and can change by a lot
    # solution: noise is very very small, basically negligible (0.1%) => ignore the ammeter noise
    keithley_picoammeter_6485_relative_standard_error = 0.1
    variance = 0.01  # TODO change this to zero

    venus = VENUS_Bayesian_Optimization(var_bounds, SEED, wait_time, sample_time)

    optimizer = bayes_opt.BayesianOptimization(
        f=venus.objective_function,
        random_state=venus.rng,
        pbounds=var_bounds,
        verbose=1,
    )

    optimizer.maximize(init_points=5, n_iter=30, kappa=4.2, alpha=variance)

    print(
        "Best result: {}; f(x) = {}.".format(
            optimizer.max["params"], optimizer.max["target"]
        )
    )
