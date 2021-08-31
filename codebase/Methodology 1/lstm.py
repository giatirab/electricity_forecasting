import torch as t
import torch.nn.functional as F
from torch import nn

# FIXME Transform tensor to Double
class LSTM(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, dropout: float
    ):  # hidden_size = loop_size

        super().__init__()  # importa tutti i metodi e attributi di nn.Module e inizializza un istanza sua

        self.LSTM = nn.LSTM(input_size=input_size, hidden_size=hidden_size)
        self.linear = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: t.Tensor) -> t.Tensor:

        x = x.view(
            -1, x.shape[1], t.prod(t.tensor(x.shape[2:]))
        )  # fa reshape del tensore da [10, 20, 11, 2] a [10, 20, 22]
        print(f"{x.shape=}")
        x, _ = self.LSTM(x)  # LSTM ritorna una tupla(a, (last_canal, last_hidden))
        x = self.dropout(x)  # IT WAS x = self.dropout(x.flatten())
        print(f"{x.shape=}")
        predicted_value = self.linear(x)  # return a unique value
        predicted_value = self.dropout(predicted_value)
        print(predicted_value.shape)

        return predicted_value
