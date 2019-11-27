"""Object-oriented approach to signal chain systematics."""

# decorators?
class SignalChain(object):
    # systematics subclass?
    def __init__(self, *args, **kwargs):
        self.models = {} # keep interpolation-like objects here
        self.seeds = {} # set of seeds per model?

    def apply(self, vis, effect):
        return vis * effect # something like this
    
    def _gen_phase(self, *args, **kwargs):
        # generate random phases
        return

# decorators?
class Gains(SignalChain):
    # do we even want the SignalChain class?
    def __init__(self, *args, **kwargs):
        # do we have a models dictionary here?
        super().__init__(*args, **kwargs)

    def bandpass(self, *args, **kwargs):
        # simulate bandpass gains
        return

    def reflection(self, *args, **kwargs):
        # simulate reflections
        return

    def apply(self, vis, model):
        # calculate effect from given model, apply to vis
        effect = getattr(self, model)()
        return super().apply(self, vis, effect)

# decorators?
class CrossTalk(SignalChain):
    # subclass of SignalChain? subclass of Gains?
    def __init__(self, *args, **kwargs):
        # do we want a models dictionary here? do we want one anywhere?
        super().__init__(*args, **kwargs)

    # make these properties?
    def white_noise(self, *args, **kwargs):
        # quick and dirty, no vis needed
        return

    def cross_coupling(self, *args, **kwargs):
        # more sophisticated, need autocorrelation visibility
        return

    def apply(self, vis, model):
        # calculate effect from given model, apply to vis
        effect = getattr(self, model)() # repeated code; can we tidy it up?
        return super().apply(vis, effect)
