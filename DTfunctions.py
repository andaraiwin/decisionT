from typing import List, Any
from collections import Counter
import math

def entropy(class_probabilities: List[float]) -> float:
    """Given a list of class probabilities, calculate the entropy"""
    return  sum(-p * math.log(p, 2)
                for p in class_probabilities
                if p > 0)

def class_probabilities(labels: List[Any]) -> List[Any]:
    total_count = len(labels)
    return [count/total_count for count in Counter(labels).values()]

def data_entropy(labels: List[Any]) -> float:
    return entropy(class_probabilities(labels))

def partition_entropy(subsets: List[List[Any]]) -> float:
    """"Returns the entropy from this partition of data into subsets"""
    total_count = sum(len(subsets) for subsets in subsets)

    return sum(data_entropy(subset) * (len(subset)/total_count) for subset in subsets)

from typing import Dict, TypeVar
from collections import defaultdict

T = TypeVar('T') # generic type for inputs

def partition_by(inputs: List[T], attribute: str) -> Dict[Any, List[T]]:
    """Partition the inputs into lists based on the specified attribute"""
    partitions: Dict[Any, List[T]] = defaultdict(list)
    for input in inputs:
        key = getattr(input, attribute)
        partitions[key].append(input)

    return partitions

def partition_entropy_by(inputs: List[Any],
                         attribute: str,
                         label_attribute: str) -> float:
    """Compute the entropy corresponding to the given partition"""
    # Partition consist of our inputs
    partitions = partition_by(inputs, attribute)

    # but partition_entropy needs just the class labels
    labels = [[getattr(input, label_attribute) for input in partition]
              for partition in partitions.values()]

    return partition_entropy(labels)


#%%
