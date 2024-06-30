import regex as re
from dataclasses import dataclass
from typing import List, Tuple, Dict, DefaultDict
from collections import defaultdict, OrderedDict
import heapq

def get_regex_pattern() -> str: 
    return r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def pre_tokenize(pattern:str, filename:str, endLine:bool=False) -> str:
    with open(filename, "r") as F:
        if not endLine:
            pre_tokenized_output:str = "".join(re.findall(pattern, "".join(line for line in F.read().splitlines())))
        else:
            pre_tokenized_output:str = "".join(re.findall(pattern, F.read()))
    return pre_tokenized_output


@dataclass(frozen=True)
class BPETokenizerParams:
    vocab: Dict[int, bytes]            
    merges: Dict[Tuple[int, int], int]  

def merge(indices: List[int], pair: Tuple[(int, int)], new_index: int, locations:Dict[int, list]) -> Tuple[List[int], Dict[int, int]]:
    """Return `indices`, but with all instances of `pair` replaced with `new_index`."""
    new_indices: list = []
    last_index: int = 0
    locations[new_index] = []  #ordered dict intializing new index
    locations_list: list = locations.pop(pair[0])
    new_locations: list = []

    for location in locations_list:
        new_indices.extend(indices[last_index:location])
        if location + 1 < len(indices) and (indices[location], indices[location + 1]) == pair:
            new_indices.append(new_index)
            locations[new_index].append(len(new_indices) - 1)

            #handle the location of the second item in the pair
            locations[pair[1]].remove(location + 1)
            if not locations[pair[1]]:
                locations.pop(pair[1])
        else:
            new_locations.append(location)
        last_index = location + 2
    new_indices.extend(indices[last_index:])

    if new_locations: 
        locations[pair[0]] = new_locations
    
    return new_indices, locations

def train_BPE(input_path:str, vocab_size:int, special_tokens:List[str]) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
        pre_tokenized = pre_tokenize(get_regex_pattern(), input_path)
        # with open(input_path, "r") as F:
        #     pre_tokenized = F.read().encode("utf-8")
        pre_tokenized = pre_tokenized.encode("utf-8")
        location_indices = OrderedDict()  #Ordered so we can use it as LRU cache when we hit vocab limit
        indices: list = []
        merges: List[Tuple[bytes, bytes]] = []
        vocab: Dict[int, bytes] = {x:bytes([x]) for x in range(256)}
        
        for i in range(len(special_tokens)):
            vocab[256 + i] = bytes(special_tokens[i].encode("utf-8"))

        for i in range(len(pre_tokenized)):
            index = int(pre_tokenized[i])
            indices.append(index)
            not_new_slice: list = location_indices.get(index, [])
            if not_new_slice:
                location_indices[index].append(i)
            else:
                not_new_slice.append(i)
                location_indices[index] = not_new_slice

        start_index = len(vocab)
        num_merges = 244
        for i in range(num_merges):
            counts: DefaultDict[Tuple[int, int], int] = defaultdict(int)

            for pair in zip(indices, indices[1:]):
                counts[pair] += 1

            #most common pair maybe need to add ordering logic in case of ties
            if not counts:
                break
            pair = max(counts, key=counts.get) 
            new_index = start_index + i
            merges.append((vocab[pair[0]], vocab[pair[1]]))
            vocab[new_index] = vocab[pair[0]] + vocab[pair[1]]

            indices, location_indices = merge(indices, pair, new_index, location_indices)

            # if len(vocab) > vocab_size:
            #     vocab.pop(location_indices.popitem(last=False)[0]) # LRU removal scheme
        
        return vocab, merges
class BPETokenizer():
    def __init__(self, input_path:str, vocab_size:int, special_tokens:List[str]) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
        pass

    def encode(self, text: str) -> List[int]:
        pre_tokenized = pre_tokenize(get_regex_pattern(), self.input_path)
        # Note: this is a very slow implementation
        for pair, new_index in self.params.merges.items():
            indices = merge(indices, pair, new_index)
        return indices

    def decode(self, indices: List[int]) -> str:
        bytes_list = list(map(self.vocab.get, indices))
        text = b"".join(bytes_list).decode("utf-8")
        return text


pattern = get_regex_pattern()
print(pre_tokenize(pattern, "test.txt"))

