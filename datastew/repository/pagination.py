from typing import Generic, List, TypeVar

from datastew.repository.model import Concept, Terminology, Mapping

T = TypeVar("T", Concept, Terminology, Mapping)


class Page(Generic[T]):
    """
    A generic class to represent a paginated set of results restricted to certain types.

    Attributes:
        items (List[T]): The list of retrieved objects.
        limit (int): The number of items per page.
        offset (int): The starting offset for the items.
        total_count (int): The total number of objects in the collection.
    """

    def __init__(self, items: List[T], limit: int, offset: int, total_count: int):
        self.items = items
        self.limit = limit
        self.offset = offset
        self.total_count = total_count

    def has_next_page(self) -> bool:
        """Check if there is a next page of results."""
        return self.offset + self.limit < self.total_count

    def has_previous_page(self) -> bool:
        """Check if there is a previous page of results."""
        return self.offset > 0

    def __repr__(self):
        return (
            f"Page(offset={self.offset}, limit={self.limit}, "
            f"total_count={self.total_count}, items_count={len(self.items)})"
        )
