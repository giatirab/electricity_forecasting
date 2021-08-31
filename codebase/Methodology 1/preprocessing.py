import numpy as np
import pandas as pd
import os
from datetime import timedelta, datetime
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import logging
import time
import torch as t
import lstm
import json


def plot_multivariate(series, list_variable_names):
    bias = 5
    for i, row in enumerate(series):
        plt.legend(list_variable_names, bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.plot(np.arange(len(row)), row + bias * i)

    plt.savefig("plots/normalised_data.png")
    plt.show()


def plot_stft(stft):
    # x = seizure_stft[0,0,0,:,0]
    # plt.bar(t.arange(len(x)), x)
    # plt.title('STFT')

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    for i in range(10):
        print(stft.shape)
        ys = stft[0, i, :]
        xs = np.arange(len(ys))
        # Plot the bar graph given by xs and ys on the plane y=k with 80% opacity.
        ax.bar(xs, ys, zs=i, zdir="y", alpha=0.8)

    ax.set_xlabel("frequency")
    ax.set_ylabel("time")
    ax.set_zlabel("contribution")
    plt.title("STFT")
    plt.show()

    plt.savefig("plots/stft.png")


def preprocessing(
    data_folder: str,
    target_variable: str,
    aug_window_size: int,
    aug_shift: int,
    stft_window_size: int,
    stft_shift: int,
    days_to_forecast: int,
    train_test_ratio: float,
    preprocessed_data_folder: str,
):

    assert (
        aug_shift < aug_window_size
    ), "Data Augmentation settings error - Shift amount must be smaller than window size!"
    assert (
        stft_shift < stft_window_size
    ), "STFT settings error - Shift amount must be smaller than window size!"

    logging.info("Reading files..")
    files = [
        f for f in os.listdir(data_folder) if f.endswith("csv")
    ]  # all files into data_folder but ignore DS_Store file

    list_variable_names: list[str] = []
    list_ts = []
    for file in files:

        assert (
            file.split(".")[-1] == "csv"
        ), f"File {file} is not valid! Please use only .csv files."
        df = pd.read_csv(os.path.join(data_folder, file))
        df.dropna(inplace=True, axis=1, how="all")  # delete all cols with NaNs
        df.dropna(inplace=True, axis=0)  # delete rows with NaNs
        if not df["Date"].is_unique:
            df = df.groupby(["Date"]).mean().reset_index()
        dates = df["Date"].values
        df.drop(["Date"], axis=1, inplace=True)
        dates = np.array(
            [
                datetime.strptime(
                    f"{date} 00:00:00" if ":" not in date else date,
                    "%Y-%m-%d %H:%M:%S",
                )
                for date in dates
            ]
        )  # FORMATTING/PARSING some date columns have hour precision
        sorting_indexes = np.argsort(dates)  # ascending ordering
        dates = dates[sorting_indexes]  # ascending ordering
        for col_name in df.columns:
            variable_name = file.strip(".csv") + "_" + col_name

            if df[col_name].dtype != np.number:
                values = df[col_name].astype(str).str.replace(",", ".").values
            else:
                values = df[col_name].values

            values = values[sorting_indexes].astype(
                float
            )  # same ascending order of dates
            if np.sum(sorting_indexes == np.arange(len(dates))) == len(sorting_indexes):
                print(
                    f"Attention!, dates for file {file} are not in the right order, I am going to sort them automatically."
                )  # python alert for sorting dates order

            list_variable_names.append(variable_name)
            list_ts.append((dates, values, file))

    target_index = list_variable_names.index(target_variable)

    logging.info("Initial processing completed.")

    start_date, end_date, min_date_diff = (
        datetime.fromtimestamp(0),
        datetime.now(),
        np.float("inf"),
    )  # initialise a inf large, inf small and inf large and replace after, all variabvles will have the same 3 variables in common

    for ts in list_ts:
        start_date_ts = ts[0][0]
        end_date_ts = ts[0][-1]
        min_date_diff_ts = (ts[0][1] - start_date_ts).total_seconds()
        print(f"{ts[2]}, {ts[0][1]}, {start_date_ts}, {min_date_diff_ts}")
        # min_date_diff_ts = min(
        #    [ts[0][i+1] - ts[0][i] for i in range(len(ts[0]) - 1)]
        # ) # TODO
        if start_date_ts > start_date:
            start_date = start_date_ts  # override, take the most recent one
        if end_date_ts < end_date:
            end_date = end_date_ts  # override, take the most outdated one
        if min_date_diff_ts < min_date_diff:
            min_date_diff = min_date_diff_ts  # can be manually chosen

    logging.info(
        f"Main variables computed: START DATE: {start_date}, END DATE: {end_date}, SAMPLING PERIOD: {min_date_diff}."
    )

    time_diff_values_ts = (
        []
    )  # list that contains lists of tuples [[(245secs, 56), (542secs, 67), (...), ...], [(...), (...), ...], [........]]

    for ts in list_ts:
        current_time_diff = []
        time_diff_values_ts.append(current_time_diff)
        ts_mask = ts[0].astype("datetime64")
        mask = (ts[0] >= start_date) & (
            ts[0] < end_date
        )  # array of Booleans, masking technique
        ts = (ts[0][mask], ts[1][mask])
        ts[0][0] = start_date
        ts[0][-1] = end_date
        for date, value in zip(*ts):  # tuple date-value
            time_diff = (date - start_date).total_seconds()
            if time_diff == 0:
                pass
            current_time_diff.append((time_diff, value))

    max_x = (
        end_date - start_date
    ).total_seconds()  # period covered in this program , converted in seconds

    xs = np.linspace(
        0, max_x, int(max_x // min_date_diff)
    )  # common dates column across all time series
    print(max_x)
    logging.info("New date index computed successfully.")
    array_to_export = []
    for ts_tuple in time_diff_values_ts:
        new_ts_tuple = np.array(ts_tuple).T
        if new_ts_tuple.dtype == np.float64:
            # spline needs values with certain shape --> get 2 vectors, one for the time passed, one for the ts values
            spline = interp1d(new_ts_tuple[0], new_ts_tuple[1], kind="cubic")
            ys = spline(xs)  # values with correct frequency
            array_to_export.append(ys)

    date_index = [
        start_date + timedelta(days=x) for x in range((end_date - start_date).days)
    ]

    array_to_export = pd.DataFrame(data=array_to_export).T
    array_to_export.columns = list_variable_names
    array_to_export.index = date_index
    array_to_export.to_pickle("cleaned_data.pickle", compression="gzip")
    array_to_export.to_csv("cleaned_data.csv")
    array_to_export = array_to_export.T
    logging.info("Saved cleaned values in .csv file.")

    array = array_to_export.values

    mean_array = np.mean(array, axis=1)
    std_array = np.std(array, axis=1)
    normalised_array = ((array.T - mean_array) / std_array).T  # z-score normalisation

    arr = pd.DataFrame(data=normalised_array).T
    arr.columns = list_variable_names
    arr.index = date_index
    arr.to_pickle("normalised_clean_data.pickle", compression="gzip")
    arr.to_csv("normalised_clean_data.csv")
    logging.info("Saved normalised values in .csv file.")

    # plot_multivariate(normalised_array, list_variable_names)
    print(normalised_array.shape)
    logging.info("Data normalisation successfully completed.")
    data = data_augment(normalised_array, window_size=aug_window_size, shift=aug_shift)
    print(data.shape)
    labels = label_windows(
        data=data,
        shift=aug_shift,
        window_size=aug_window_size,
        days_to_forecast=days_to_forecast,
        target_index=target_index,
    )
    print(len(labels))
    data = data[: len(labels), :, :]

    # NEXT
    new_data = tuple(
        short_term_fourier_transform(x, window_size=stft_window_size, shift=stft_shift)
        for x in data
    )  # array of shape sub_windows 20/5, num_variabes, extracted Fourier frequencies, int+complex
    print(new_data[0].shape)
    plot_stft(new_data[0])

    new_data = np.array(new_data)
    # new_data must be tranformed into tenor before LSTM step
    new_data = t.from_numpy(new_data)
    print(f"{new_data.shape=}")
    train_set, test_set, train_set_labels, test_set_labels = train_test_split(
        new_data, labels, train_test_ratio
    )

    # save train and test datasets into preprocessed folder
    if not os.path.exists(preprocessed_data_folder):
        os.mkdir(preprocessed_data_folder)
    t.save(train_set, os.path.join(preprocessed_data_folder, "train_data.pkl"))
    t.save(test_set, os.path.join(preprocessed_data_folder, "test_data.pkl"))
    t.save(
        train_set_labels,
        os.path.join(preprocessed_data_folder, "train_labels_data.pkl"),
    )
    t.save(
        test_set_labels, os.path.join(preprocessed_data_folder, "test_labels_data.pkl")
    )

    with open(os.path.join(preprocessed_data_folder, "variable_names"), "w") as f:
        json.dump(list_variable_names, f, indent=5)

    return new_data

    print(
        f"This is the name we want: {list_variable_names.index('italy_electricity_close_price_GMERTRB')}"
    )


def train_test_split(
    data: t.Tensor, labels: t.Tensor, train_test_ratio: float
) -> tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor]:

    rand_indices = t.randperm(
        data.shape[0]
    )  # returns a range from 0 to len(data.shape[0]) with random order
    train_set, test_set = (
        data[rand_indices[: int(np.floor(train_test_ratio * data.shape[0]))]],
        data[rand_indices[int(np.floor(train_test_ratio * data.shape[0])) :]],
    )

    train_set_labels, test_set_labels = (
        labels[rand_indices[: int(np.floor(train_test_ratio * labels.shape[0]))]],
        labels[rand_indices[int(np.floor(train_test_ratio * labels.shape[0])) :]],
    )
    print(f"{train_set.shape=} {test_set.shape=}")
    return (train_set, test_set, train_set_labels, test_set_labels)


def data_augment(data: np.ndarray, window_size: int, shift: int) -> np.ndarray:

    num_windows = (data.shape[1] - window_size) // shift + 1  # len examples
    windows = np.array(
        tuple(data[:, i * shift : i * shift + window_size] for i in range(num_windows))
    )  # tuple comprehension is faster than for cycle

    return windows


def short_term_fourier_transform(
    data: np.ndarray, window_size: int, shift: int
) -> np.ndarray:

    num_windows = data.shape[1] // window_size
    windows = tuple(
        data[:, i * shift : i * shift + window_size] for i in range(num_windows)
    )  # tuple comprehension is faster than for cycle
    fft_windows = np.array(
        tuple(np.fft.rfft(window) for window in windows)
    )  # apply FFT on each sub_window, made of complex numbers (real + imaginery)

    real_array = np.real(fft_windows)
    complex_array = np.imag(fft_windows)

    array = np.array((real_array, complex_array))
    result = array.transpose(
        1, 2, 3, 0
    )  # this and the above line can be made a unique command in pytorch

    return result.squeeze()  # removes 1 dimensionality


# TODO FOR NEXT TIME
def label_windows(
    data: np.ndarray,
    shift: int,
    window_size: int,
    target_index: int,
    days_to_forecast: int = 365,
) -> t.Tensor:

    num_labels_to_assign = data.shape[0] - int(np.ceil(days_to_forecast / shift))
    print(num_labels_to_assign)
    # slicing

    list_labels: list[float] = []
    for i in range(num_labels_to_assign):
        starting_window = data[i]
        target_window_index = i + days_to_forecast // shift
        target_window = data[target_window_index]
        # print(f"{target_window=}, {target_index=}")
        avg_elec_price_target_window = np.mean(
            (data[target_window_index, target_index, :])
        )  # media dei prezzi gia normalizzati dell elettricita per quella target window
        list_labels.append(avg_elec_price_target_window)

    return t.tensor(list_labels)


def from_days_to_window(day: int, shift: int) -> int:
    return day // shift


def test_labels(data: np.ndarray, list_labels: list[float]) -> bool:
    """Check if the label value is actually the mean for that window where it lies."""
    # TODO
    pass
