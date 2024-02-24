from abc import ABC, abstractmethod


class BaseRepository(ABC):
    @abstractmethod
    def store(self, model_object_instance):
        """Store a single model object instance."""
        pass

    @abstractmethod
    def store_all(self, model_object_instances):
        """Store multiple model object instances."""
        pass

    @abstractmethod
    def get_closest_mappings(self, embedding, limit=5):
        """Get the closest mappings based on embedding."""
        pass

    @abstractmethod
    def shut_down(self):
        """Shut down the repository."""
        pass
