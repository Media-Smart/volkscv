from abc import abstractmethod


class BaseAssigner:

    @abstractmethod
    def assign(self, *args, **kwargs):
        pass
