from typing import List, Dict, Set, Any, Union

Item = Any
Hash = Union[int, float, str]

class FastQueue:   
    items: List[Any]
    item_lookup: Dict[Hash, Any]
    hashes: Set[Hash]
    item_count: int

    def __init__(self) -> None:
        self.items = []
        self.item_lookup = {}
        self.hashes = set() 
        self.item_count = 0

    def add(self, item: Any, hash: Any) -> None:
        if hash in self.hashes:
            raise ValueError(f"Hash has already been added.")
        
        self.hashes.add(hash)
        self.item_lookup[hash] = self.item_count
        self.items.append(item)
        self.item_count += 1   
    
    def consume(self) -> Item:
        pass
    
    def consume_all(self) -> List[Item]:
        pass

    def flush(self, num_items: int) -> None:
        if num_items > self.num_items:
            raise IndexError(f"Too many items to flush, queue size is {self.num_items}.")
        
    def clear(self) -> None:
        self.items.clear()
        self.item_lookup.clear()
        self.hashes.clear()
        self.num_items = 0
    