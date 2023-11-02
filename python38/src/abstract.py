from abc import ABC, abstractmethod

class Abstract(ABC):

    @abstractmethod
    def __repr__(self) -> str:
        """Show the current class info."""
        raise NotImplementedError("Have not implemented __repr__ yet!")

    def show(self) -> None:
        print(self)

    def attributes(self) -> None:
        # Using the vars() function
        print(vars(self))