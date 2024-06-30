import regex as re
from dataclasses import dataclass
from typing import List, Tuple, Dict, OrderedDict

def get_regex_pattern() -> str: 
    return r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def pre_tokenize(pattern:str, filename:str, endLine:bool=False) -> str:
    with open(filename, "r") as F:
        if not endLine:
            pre_tokenized_output:str = "".join(re.findall(pattern, "".join(line for line in F.read().splitlines())))
        else:
            pre_tokenized_output:str = "".join(re.findall(pattern, F.read()))
        pre_tokenized_output = list(map(int, pre_tokenized_output.encode("utf-8")))
    return pre_tokenized_output


@dataclass(frozen=True)
class BPETokenizerParams:
    vocab: Dict[int, bytes]            
    merges: Dict[Tuple[int, int], int]  

def merge(indices: List[int], pair: Tuple[(int, int)], new_index: int) -> List[int]:
    """Return `indices`, but with all instances of `pair` replaced with `new_index`."""
    new_indices = []
    i = 0
    while i < len(indices):
        if i + 1 < len(indices) and indices[i] == pair[0] and indices[i + 1] == pair[1]:
            new_indices.append(new_index)
            i += 2
        else:
            new_indices.append(indices[i])
            i += 1
    return new_indices

class BPETokenizer():
    def __init__(self, input_path:str, vocab_size:int, special_tokens:List[str]) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
        self.pre_tokenized = pre_tokenize(get_regex_pattern(), input_path)
        self.pre_tokenized = list
        self.vocab:Dict[] = {}
        self.input_path = input_path

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