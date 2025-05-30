
import torch.nn as nn

class AdapterPipe:
    # contains the helper methods and the array of adapters on the pipeline

    def __init__(self, adapters=None):
        self.adapters = adapters if adapters is not None else []
        # snap together as many as you want and they will be called in order

    def add_adapter(self, adapter):
        if not isinstance(adapter, nn.Module):
            raise TypeError("Adapter must be an instance of nn.Module.")
        self.adapters.append(adapter)

    def forward(self, input1, input2):
        ...


