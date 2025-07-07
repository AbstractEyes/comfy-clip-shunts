
import torch.nn as nn

class ShuntAdapterRepresentation:
    pass

class AdapterPipe:
    ...
    # accepts any number of adapters;
    # the adapter is grouped under it's encoder target pair, which is the pair of expected models
    # most adapters can ATTEMPT to use another encoder if the size matches, but this is not guaranteed to work.
    # # loaded_encoders: {} # # list of loaded encoders that any currently loaded adapter can use
    # # strict: if True, the adapter will only be used if the encoder matches the expected encoder

    def __init__(self):
        # pipes self clean after each run, persistence does not exist
        self.loaded_encoders = {} # all of the encoders loaded and passed through this pipe

        # these apply to all adapters rather than just individuals
        self.strict = False  # if True, the adapter will only be used if the encoder matches the expected encoder
        self.quiet_fail = False  # if True, the pipe will ignore any errors during processing and continue no matter what
        self.crash_on_fail = True  # if True, the pipe will raise an error on any failure during processing

        # loud fails and ignoring errors is a method to large scale adapter usage, which allows more flexibility

    def add_adapter(self, adapter, expected_encoder="any", strict=False, quiet_fail=False, crash_on_fail=False):
        """
        Adds an adapter to the pipe.
        :param adapter: The adapter to add.
        :param expected_encoder: The encoder that the adapter expects.
        :param strict: If True, the adapter will only be used if the encoder matches the expected encoder.
        :param quiet_fail: If True, the pipe will ignore any errors during processing.
        :param crash_on_fail: If True, the pipe will raise an error on any failure during processing.
        """
        self.loaded_encoders[expected_encoder] = adapter
        self.strict = strict
        self.quiet_fail = quiet_fail
        self.crash_on_fail = crash_on_fail

    #def prepare


class EncoderPipe(AdapterPipe):
    ...


class ConditioningPipe(AdapterPipe):
    ...
