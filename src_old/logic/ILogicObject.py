import abc


class ILogicObject(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def update(self):
        pass