from argparse import ArgumentParser
from preprocessing import preprocessing
from lstm_manager import LSTMManager
import logging
from lstm_manager import LSTMManager


# FIXME fix the 1 dimension in fourier
# TODO alert on csv dates
# TODO preprocess args into main.py as args
# TODO min_date_diff flexible
# TODO segmentation
# TODO fourier transform STFT
# TODO kalman filtering
# TODO non-linear correlation


def main():

    parser = ArgumentParser(
        description="This is an intro to this program and how to use it.."
    )
    parser.add_argument(
        "action", type=str, choices=["preprocess", "train", "test"]
    )  # the user decises if you start from preprocessing or from training directly
    parser.add_argument(
        "-d",  #  quick version to pass dropout
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout is an optional argument that deactivates some neurons etc..",
    )  # -- means it is an optional parameter
    parser.add_argument(
        "-dd",
        "--data-dir",
        type=str,
        default="data",
        help="This is the location of your csv files..",
    )
    parser.add_argument(
        "-pdf",
        "--preprocessed-data-folder",
        type=str,
        default="preprocessed_data",
        help="This is the location where to save your train and test datasets.",
    )
    parser.add_argument(
        "-tv",
        "--target-variable",
        type=str,
        default="italy_electricity_close_price_GMERTRB",
        help="Please pick your target variable..",
    )  # In terminal, arguments do not use _ but - by convention
    parser.add_argument(
        "-r",
        "--train-test-ratio",
        type=float,
        default=0.75,
        help="Please select your train and test ratio split..",
    )  # In terminal, arguments do not use _ but - by convention
    parser.add_argument(
        "-lr",
        "--lr",
        type=float,
        default=0.001,
        help="Please select your LSTM learning rate.",
    )  # In terminal, arguments do not use _ but - by convention
    parser.add_argument(
        "-nep",
        "--num_epochs",
        type=int,
        default=1,
        help="Please choose your number of epochs size.",
    )  # In terminal, arguments do not use _ but - by convention
    parser.add_argument(
        "-trbs",
        "--train-batch-size",
        type=int,
        default=10,
        help="Please choose your train batch size.",
    )  # In terminal, arguments do not use _ but - by convention
    parser.add_argument(
        "-tebs",
        "--test-batch-size",
        type=int,
        default=100,
        help="Please choose your test batch size.",
    )  # In terminal, arguments do not use _ but - by convention
    parser.add_argument(
        "-agw",
        "--aug-window-size",
        type=int,
        default=20,
        help="Please select your data aumentation window size.",
    )  # In terminal, arguments do not use _ but - by convention
    parser.add_argument(
        "-ags",
        "--aug-shift",
        type=int,
        default=2,
        help="Please select your data aumentation shift amount.",
    )  # In terminal, arguments do not use _ but - by convention
    parser.add_argument(
        "-stftw",
        "--stft-window-size",
        type=int,
        default=20,
        help="Please select your STFT window size.",
    )  # In terminal, arguments do not use _ but - by convention
    parser.add_argument(
        "-stfts",
        "--stft-shift",
        type=int,
        default=1,
        help="Please select your STFT shift amount.",
    )  # In terminal, arguments do not use _ but - by convention
    parser.add_argument(
        "-df",
        "--days-to-forecast",
        type=int,
        default=365,
        help="Please select the number of days to forecast.",
    )  # In terminal, arguments do not use _ but - by convention

    args = parser.parse_args()

    # Some testing on the input variables
    assert args.train_test_ratio <= 1, "Your ratio is not valid!"
    assert args.lr > 0, "The learning rate should be positive!"
    assert (
        args.aug_window_size > args.aug_shift
    ), "Data Augmentation error - Your window size cannot be larger than your shift"
    assert (
        args.stft_window_size > args.stft_shift
    ), "STFT error - Your window size cannot be larger than your shift"
    print(f"{args.aug_window_size=} | {args.stft_window_size=}")
    assert (
        args.aug_window_size >= args.stft_window_size
    ), "Your are trying to decompose a window larger than the available window size."

    if args.action == "preprocess":
        print("Running program from preprocessing step onwards.")
        logging.basicConfig(filename="mylog.log", level=logging.INFO)
        preprocessing(
            data_folder=args.data_dir,
            target_variable=args.target_variable,
            aug_window_size=args.aug_window_size,
            aug_shift=args.aug_shift,
            stft_window_size=args.stft_window_size,
            stft_shift=args.stft_shift,
            days_to_forecast=args.days_to_forecast,
            train_test_ratio=args.train_test_ratio,
            preprocessed_data_folder=args.preprocessed_data_folder,
        )
    if args.action == "train":
        manager = LSTMManager(
            preprocessed_data_folder=args.preprocessed_data_folder,
            lr=args.lr,
            num_epochs=args.num_epochs,
            train_batch_size=args.train_batch_size,
            test_batch_size=args.test_batch_size,
            dropout=args.dropout,
        )
        manager.train()

    if args.action == "test":
        pass
    # train()
    # test()

    print("Program completed!")


if __name__ == "__main__":
    main()
