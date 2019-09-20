import matplotlib.pyplot as plt
import math
from hmm_class import hmm
from random import random
import numpy as np

# people count data file
data_file = open("CalIt2.data", "r")
cnt_data = data_file.readlines()
data_file.close()

# event record data file
data_file = open("CalIt2.events", "r")
event_data = data_file.readlines()
data_file.close()

# possible vals of events
events = None

# possible states
states = None


def process_line(s):
    """Parse date and people count in a single input string"""
    count = int(s.strip().split(",")[3])
    date = s.strip().split(",")[1]
    return date, count


def process_raw_data():
    """
    Parses raw data
    returns:
        `data_set`: dictionary where every entry is `key` of date and `value`
            of another dictionary with `count` as people count and `event` as
            Boolean that is `True` if an event occurred on that date
        `data_list`: sequence of people counts
    """
    data_set = {}
    for idx in range(len(cnt_data) - 2):
        line = cnt_data[idx]
        date, count = process_line(line)
        if date in data_set:
            data_set[date]["count"] += count
        else:
            data_set[date] = {"count": count, "event": False}
    for line in event_data:
        date = line.split(",")[0]
        data_set[date]["event"] = True

    data_list = []
    for key in data_set.keys():
        data_list.append(data_set[key]["count"])

    return data_set, data_list


def discretize_data(data_list, data_max, data_min, num_intervals):
    """
    Divide interval `[data_min, data_max]` into `num_intervals` intervals
    and convert `data_list` elements to indices of intervals
    """
    interval = (data_max - data_min) / num_intervals

    def float_to_index(val):
        return math.floor((val - data_min) / interval)

    return list(map(float_to_index, data_list))


def norm_vec(vec):
    """
    Normalizes given vector of numbers.
    Assumes that sum of `vec` is non-zero.
    """
    new_vec = vec[:]
    s = sum(new_vec)
    for idx in range(len(new_vec)):
        new_vec[idx] /= s
        idx += 1
    return new_vec


def gen_states(num_events):
    """
    Sets global vars `events` and `states` to tuple of possible event values
    and tuple of possible state values respectively
    """
    global events, states
    events = tuple(range(num_events))
    states = []
    for day in range(7):
        for event in events:
            states.append(str(day) + str(event))
    states = tuple(states)


def convert_state_to_idx(dayOfWk, event):
    """Convert day of week, event value information to index within `states`"""
    event_idx = events.index(event)
    return len(events) * (dayOfWk % 7) + event_idx


def gen_hmm_inputs(num_intervals):
    """
    Generates inputs for HMM
    Returns:
        `states`
        `symbols`
        `start_prob`
        `trans_prob`
        `emit_prob`
    """
    num_events = len(events)
    startDayOfWk = 6  # data start from Sunday

    num_states = len(states)
    symbols = tuple(range(num_intervals))

    # init start probs
    start_prob = [0.0000001] * num_states
    idx = convert_state_to_idx(startDayOfWk, events[0])
    # High prob to Sunday, no event
    start_prob[idx:idx + num_events] = [10.0] + [0.1] * (num_events - 1)

    # init transition probs
    trans_prob = [[0.0000001 for i in range(num_states)]
                  for j in range(num_states)]
    for day in range(7):
        today_idx = convert_state_to_idx(day, events[0])
        tmr_idx = convert_state_to_idx(day + 1, events[0])
        # some prob to transitions to next day
        trans_prob[today_idx][tmr_idx:tmr_idx + num_events] = [0.1]*num_events
        # tomorrow, current and next event values are more likely
        for event_idx in range(num_events - 1):
            trans_prob[today_idx + event_idx][tmr_idx + event_idx] = 10.
            trans_prob[today_idx + event_idx][tmr_idx + event_idx + 1] = 10.
        # boundary index
        trans_prob[today_idx + num_events - 1][tmr_idx + num_events - 1] = 10.
        trans_prob[today_idx + num_events - 1][tmr_idx] = 10.

    # init emission probs
    fraction = 4  # hyperparam
    emit_prob = [[0.0000001 for i in range(num_intervals)]
                 for j in range(num_states)]
    # weekdays
    for day in range(5):
        idx = convert_state_to_idx(day, events[0])
        # assign higher probs to higher people count
        emit_prob[idx] = [0.1] * (num_intervals // fraction) + \
            [5.0] * (num_intervals - (num_intervals // fraction))
        # for some nonzero event vals
        for event_idx in range(1, num_events):
            emit_prob[idx + event_idx] = [0.1] * (num_intervals // fraction) + \
                [5.0] * (num_intervals - (num_intervals // fraction))
    # weekends
    for day in range(5, 7):
        idx = convert_state_to_idx(day, events[0])
        for event_idx in range(num_events):
            # assign higher probs to lower people count
            emit_prob[idx + event_idx] = [10.0] * (num_intervals // fraction) + \
                [0.1] * (num_intervals - (num_intervals // fraction))

    # normalize
    start_prob = norm_vec(start_prob)
    trans_prob = list(map(norm_vec, trans_prob))
    emit_prob = list(map(norm_vec, emit_prob))

    return states, symbols, start_prob, trans_prob, emit_prob


def calc_mse(list1, list2):
    """Calculates mean squared error between `list1` and `list2`"""
    N = len(list1)
    mse = 0
    for idx in range(N):
        mse += (list2[idx] - list1[idx]) ** 2
    mse /= N
    return mse


def calc_mean_abs_err(list1, list2):
    """Calculates mean of absolute error between `list1` and `list2`"""
    N = len(list1)
    mae = 0
    for idx in range(N):
        mae += abs(list2[idx] - list1[idx])
    mae /= N
    return mae


def run_exp(num_events):
    """run experiment"""
    N_train = 75                        # training seq size
    N_test = 20                         # test seq size
    start_train = 0                     # training start index
    start_test = start_train + N_train  # test start index

    # parse data
    data_set, data_list = process_raw_data()
    train_data = data_list[start_train:start_train + N_train]
    test_data = data_list[start_test:start_test + N_test]

    # discretize data list
    num_intervals = 20
    data_max = max(data_list)
    data_min = 0
    disc_data_list = discretize_data(data_list, data_max,
                                     data_min, num_intervals)

    # x values for plotting later, discretized data lists for viterbi later
    train_x = list(range(start_train, start_train + N_train))
    disc_train_data = disc_data_list[start_train:start_train + N_train]
    test_x = list(range(start_test, start_test + N_test))
    disc_test_data = disc_data_list[start_test:start_test + N_test]

    # get inputs for HMM
    gen_states(num_events)
    states, symbols, start_prob, trans_prob, emit_prob = gen_hmm_inputs(num_intervals)
    start_prob = np.matrix(start_prob)
    trans_prob = np.matrix(trans_prob)
    emit_prob = np.matrix(emit_prob)

    # init HMM
    model = hmm(states, symbols, start_prob, trans_prob, emit_prob)

    # train HMM
    num_iters = 1000
    new_emit_prob, new_trans_prob, new_start_prob = \
        model.train_hmm([disc_train_data], num_iters, [1])
    new_emit_prob = np.asarray(new_emit_prob).tolist()
    new_trans_prob = np.asarray(new_trans_prob).tolist()
    new_start_prob = np.asarray(new_start_prob).tolist()

    # viterbi on training obs seq
    hidden_state_seq = model.viterbi(disc_train_data)
    log_probs = model.log_prob([disc_train_data], [1])

    # # This code was used when I needed to plot existence of events against
    # # date
    # z1 = [0] * len(data_set.keys()) # existence of events
    # x1 = list(range(len(z1)))
    # idx = 0
    # for key in data_set.keys():
    #     if data_set[key]["event"]:
    #         z1[idx] = 1
    #     idx += 1

    # length of interval
    interval = (data_max - data_min) / num_intervals

    def hidden_state_to_entry(s):
        """Convert hidden state to observation value"""
        state_idx = states.index(s)
        emit_sum = 0
        for emit_idx in range(num_intervals):
            emit_sum += new_emit_prob[state_idx][emit_idx] * emit_idx
        return data_min + interval * emit_sum

    # reconstructed observation seq
    train_rec = list(map(hidden_state_to_entry, hidden_state_seq))

    # compare reconstructed obs seq to actual obs seq
    plt.plot(train_x, train_data, "r", label="Actual")
    plt.plot(train_x, train_rec, "b", label="Predicted")
    plt.xlabel("Days passed")
    plt.ylabel("Flow of people")
    plt.suptitle("Prediction of People Count Observed on Training Sequence")
    plt.legend(loc="upper left")
    plt.show()

    # calculate MSE, mean of abs err
    train_mse = calc_mse(train_data, train_rec)
    train_mae = calc_mean_abs_err(train_data, train_rec)

    # viterbi on test observation seq
    test_viterbi_res = model.viterbi(disc_train_data + disc_test_data)
    test_hidden_state_seq = test_viterbi_res[start_test:start_test + N_test]
    test_log_probs = model.log_prob([disc_train_data + disc_test_data], [1])

    # reconstructed observation seq
    test_rec = list(map(hidden_state_to_entry, test_hidden_state_seq))

    # compare reconstructed obs seq to actual obs seq
    plt.plot(test_x, test_data, "r", label="Actual")
    plt.plot(test_x, test_rec, "b", label="Predicted")
    plt.xlabel("Days passed")
    plt.ylabel("Flow of people")
    plt.suptitle("Prediction of People Count Observed on Test Sequence")
    plt.legend(loc="upper left")
    plt.show()

    # calculate MSE, mean of abs err
    test_mse = calc_mse(test_data, test_rec)
    test_mae = calc_mean_abs_err(test_data, test_rec)

    return (train_mse, train_mae, log_probs, test_mse, test_mae,
            test_log_probs)


if __name__ == "__main__":
    for num_events in [5, 10, 20, 30, 40, 50, 60, 70, 80, 90,
                       100, 110, 120, 130, 140, 150]:
        print("{}: {}".format(num_events, run_exp(num_events)))
