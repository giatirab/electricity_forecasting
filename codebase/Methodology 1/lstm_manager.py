from lstm import LSTM
import torch as t
from torch.utils.tensorboard import SummaryWriter
import tensorboard
import os


class Batch:
    def __init__(self, windows: t.Tensor, labels: t.Tensor):

        self.windows = windows
        self.labels = labels


class LSTMManager:
    def __init__(
        self,
        preprocessed_data_folder: str,
        lr: float,
        num_epochs: int,
        train_batch_size: int,
        test_batch_size: int,
        dropout: float,
    ):

        self.lr = lr  # learning_rate
        self.num_epochs = num_epochs
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.loss_function = t.nn.MSELoss()
        self.writer = SummaryWriter("runs")
        self.train_set = t.load(
            os.path.join(preprocessed_data_folder, "train_data.pkl")
        )
        self.test_set = t.load(os.path.join(preprocessed_data_folder, "test_data.pkl"))
        self.train_set_labels = t.load(
            os.path.join(preprocessed_data_folder, "train_labels_data.pkl"),
        )
        self.test_set_labels = t.load(
            os.path.join(preprocessed_data_folder, "test_labels_data.pkl"),
        )
        input_size = t.prod(t.tensor(self.train_set.shape[2:]))
        hidden_size = self.train_set.shape[1]
        print(f"{input_size=}, {hidden_size=}")
        self.model = LSTM(
            input_size=input_size, hidden_size=hidden_size, dropout=dropout
        )
        self.optimizer = t.optim.Adam(self.model.parameters(), lr=self.lr)
        print(
            f"{self.train_set.shape=}, {self.test_set.shape=}, {self.train_set_labels.shape=}, {self.test_set_labels.shape=}"
        )

    def train(self):
        """Docstring here."""
        # Divide data in batches
        train_batches, _ = self.__batchify(self.train_batch_size)

        cum_loss = t.tensor(0, dtype=t.float64)
        i = 0
        for epoch in range(self.num_epochs):
            for (
                tb
            ) in (
                train_batches
            ):  # tb of shape ([10, 20, 11, 2], [10]), as the train_bacth = 10
                print(tb.windows.shape)
                print(tb.labels.shape)
                i += 1
                self.model.float().train()
                self.optimizer.zero_grad()  # azzera i gradienti dei tensori
                predicted_values = self.model(tb.windows.float())
                loss = self.loss_function(tb.labels, predicted_values)
                print(f"{loss=}")
                cum_loss += loss
                self.writer.add_scalar(
                    tag="batch_loss", scalar_value=loss, global_step=i
                )
                self.optimizer.step()  # esegue la backpropagation
                self.test(epoch)  # test at every epoch
            self.writer.add_scalar(
                tag="epoch_loss", scalar_value=cum_loss, global_step=epoch
            )

        print("Training completed successfully.")

    def test(self, epoch: int):
        """Docstring here."""
        # Divide data in batches
        _, test_batches = self.__batchify(self.test_batch_size)

        cum_loss = t.tensor(0)
        i = 0
        with t.no_grad():  # non fare il calcolo del gradiente
            self.model.eval()
            for tb in test_batches:
                i += 1
                predicted_values = self.model(tb.windows)  # contains fropout layer
                loss = self.loss_function(tb.labels, predicted_values)
                cum_loss += loss
                self.writer.add_scalar(
                    tag="batch_loss", scalar_value=loss, global_step=i
                )
                self.optimizer.step()  # esegue la backpropagation
            self.writer.add_scalar(
                tag="test_loss", scalar_value=cum_loss, global_step=epoch
            )
        if (
            epoch == self.num_epochs - 1
        ):  # just print this when testing on the whole test set
            print("Testing completed successfully.")

    def __batchify(
        self, batch_size: int
    ) -> tuple[
        tuple[Batch, ...], tuple[Batch, ...]
    ]:  # _ protected, __ private, con 2 __ funziona

        train = t.split(self.train_set, batch_size)
        test = t.split(self.test_set, batch_size)
        train_labels = t.split(self.train_set_labels, batch_size)
        test_labels = t.split(self.test_set_labels, batch_size)

        train_batches = tuple(
            Batch(data, labels) for data, labels in zip(train, train_labels)
        )  # tuple of batches, each cointaining the data and relative labels
        test_batches = tuple(
            Batch(data, labels) for data, labels in zip(test, test_labels)
        )

        return (train_batches, test_batches)

    def __call__(self, window):  # call a n instance as function
        pass
