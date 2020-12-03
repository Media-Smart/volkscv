from ..base import Base
from ..plotter import BasePlotter, Compose, SubPlotter


class BaseProcessor(Base):
    """ Base class of processors.

    Args:
        data: Data to be processed.
    """

    def __init__(self, data):
        self.data = data
        self.processor = []

    def default_plot(self, *args, **kwargs):
        """ A default plot function. It will call the default_plot function
        of each processed processors in self.processor.
        """

        for p in self.processor:
            p_ = getattr(self, p)
            if isinstance(p_, SubPlotter):
                p_.plot()
            elif isinstance(p_, BasePlotter):
                p_.figure(p_.text, *args, **kwargs)
                p_.title(p_.text)
                p_.plot()
            elif isinstance(p_, Compose):
                if p_.flag:
                    p_.plot()
                else:
                    p_.figure(p_.text, *args, **kwargs)
                    p_.title(p_.text)
                    p_.plot()
            elif isinstance(p_, BaseProcessor):
                p_.default_plot(*args, **kwargs)
            else:
                continue

    def plot(self):
        """ Call the plot method or each processor in self.processor."""

        for p in self.processor:
            if hasattr(self, p):
                p_ = getattr(self, p)
                p_.plot()
