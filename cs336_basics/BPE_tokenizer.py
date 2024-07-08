import regex as re
from dataclasses import dataclass
from typing import List,Tuple, Dict, DefaultDict, OrderedDict as OD
from collections import defaultdict, OrderedDict
import heapq
import json

def get_regex_pattern(pattern:str) -> str: 
    reg: str = ""
    if pattern == "GPT2":
        reg = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    elif pattern == "GPT4":
        reg = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
    elif pattern == "BPE_Example":
        reg = r"""\S"""
    else:
        raise NotImplementedError
    return re.compile(reg)

def pre_tokenize(pattern:str, filename:str, endLine:bool=True) -> List[str]:
    with open(filename, "r", encoding="utf-8") as F:
        if not endLine:  # Make sure to keep each word separate. Do not concat string
            pre_tokenized_output:List[str] = re.findall(pattern, "".join(line + " " for line in F.read().splitlines()))
        else:
            pre_tokenized_output:List[str] = re.findall(pattern, F.read())
    return pre_tokenized_output

def build_pairs_and_locations(indices: List[int], counts_and_locations: Dict[Tuple[int, int], list], index: int) -> Dict[Tuple[int, int], list]:
    for pair in zip(indices, indices[1:]):
        counts_and_locations[pair] = counts_and_locations.get(pair, [0, []])
        counts_and_locations[pair][0] += 1
        counts_and_locations[pair][1].append(index)
    return counts_and_locations

def save_merges_and_vocab(merges: List[Tuple[int, int]], vocab:dict[int, int], prefix="") -> None:
        with open(f"{prefix}_merges.txt", "w") as f:
            for key, val in merges:
                f.write(f"({key} {val})\n")
        
        with open(f"{prefix}_vocab.txt", "w") as f:
            for key, val in vocab.items():
                f.write(f"({key}:{val})\n")


@dataclass(frozen=True)
class BPETokenizerParams:
    vocab: Dict[int, bytes]            
    merges: Dict[Tuple[int, int], int]  

def merge(indices: List[int], pair: Tuple[(int, int)], new_index: int) -> List[int]:
    new_indices: List[int] = []
    i: int = 0
    while i < len(indices):
        if i + 1 < len(indices) and (indices[i],indices[i+1]) == pair:
            new_indices.append(new_index)
            i += 2
        else:
            new_indices.append(indices[i])
            i += 1
    return new_indices


def train_BPE(input_path:str, vocab_size:int, special_tokens:List[str]) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
        regex:re.Pattern = get_regex_pattern("GPT4")
        pre_tokenized: List[str] = pre_tokenize(regex, input_path)
        # with open(input_path, "r") as text:
            # pre_tokenized = F.read().encode("utf-8")
        counts_and_locations: Dict[Tuple[int,int], list] = {}
        heap: List[Tuple[int, Tuple[int, int]]] = []

        vocab: Dict[int, bytes]  = {i:bytes(special_tokens[i].encode("utf-8")) for i in range(len(special_tokens))}

        indices: List[List[int]] = []
        merges: List[Tuple[bytes, bytes]] = []
        for x in range(256):
            vocab[x + len(special_tokens)] = bytes([x])

        for i in range(len(pre_tokenized)):
            word_ints: List[int] = list(pre_tokenized[i].encode("utf-8"))
            indices.append(word_ints)
            counts_and_locations = build_pairs_and_locations(word_ints, counts_and_locations, i)
                
        for key, value in counts_and_locations.items():
            heapq.heappush(heap, (-value[0], key))

        start_index = len(vocab)
        num_merges = vocab_size - start_index
        for idx in range(num_merges):
            #most common pair maybe need to add ordering logic in case of ties
            pair: Tuple[int, int]  = (0, 0)
            max_counts: int = 0
            max_counts, pair = heapq.heappop(heap)

            ties: List[Tuple[str, Tuple[int, int]]] = [("".join([vocab[x].decode("utf-8", errors="replace") for x in pair]), pair)]
            while heap[0][0] == max_counts:
                next_pair: Tuple[int, int] = heapq.heappop(heap)[1]
                ties.append(("".join([vocab[x].decode("utf-8") for x in next_pair]), next_pair))

            sorted_ties:List[Tuple[str, Tuple[int, int]]] = sorted(ties, reverse=True)
            for i in range(len(sorted_ties)):     
                if i == 0:
                    pair = sorted_ties[i][1]
                else:
                    heapq.heappush(heap, (max_counts, sorted_ties[i][1]))

            new_index = start_index + i
            merges.append((vocab[pair[0]], vocab[pair[1]]))
            vocab[new_index] = vocab[pair[0]] + vocab[pair[1]]

            indices = merge(indices, pair, new_index)

        save_merges_and_vocab(merges, vocab, prefix="taylorswift")
        
        return vocab, merges

vocab, merges = train_BPE("minbpe/tests/taylorswift.txt", 512, [])

