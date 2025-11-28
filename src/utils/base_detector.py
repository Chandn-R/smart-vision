from abc import ABC, abstractmethod

class BaseDetector(ABC):

    @abstractmethod
    def load_model(self, model_path):
        """Loads the model from a file."""
        pass

    @abstractmethod
    def predict(self, frame):
        """Runs inference on a single frame."""
        pass