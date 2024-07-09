import regex as re
from dataclasses import dataclass
from typing import List,Tuple, Dict, DefaultDict, Deque, Set, OrderedDict as OD
from collections import defaultdict, OrderedDict, deque
import heapq

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

def save_merges_and_vocab(merges: List[Tuple[bytes, bytes]], vocab:dict[int, bytes], prefix="") -> None:
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

def merge(indices: List[List[int]], pair: Tuple[(int, int)], new_index: int, counts_and_locations: Dict[Tuple[int, int], list], negatives: Set[Tuple[int, int]]) -> Tuple[List[List[int]],  Dict[Tuple[int, int], list],  List[Tuple[int, int]], Set[Tuple[int, int]]]:
    locations_to_replace = counts_and_locations.pop(pair)[1]  # get only the locations, the counts will not matter anymore since we are merging
    pairs_to_update_counts: List[Tuple[int, int]] = [ ]
    for indices_index in locations_to_replace:
        new_pair: Deque = deque([])
        word_ints = deque(indices[indices_index])
        new_indices: List[int] = []

        while word_ints:
            popped_number: int = word_ints.popleft()
            new_pair.append(popped_number)
            new_indices.append(popped_number)

            if len(new_pair) > 1 and new_pair[0] == pair[0] and new_pair[1] == pair[1]:
                if len(new_indices) > 2:
                    negatives.add((new_indices[-3], new_indices[-2]))
                if word_ints:
                    negatives.add((new_indices[-1], word_ints[0]))
                new_indices.pop()
                new_indices.pop()
                new_indices.append(new_index)
                new_pair = deque([new_index])

            if new_indices:
                pair_to_update: Tuple[int, int] = ()
                if new_indices[-1] == new_index:
                    pair_to_update: Tuple[int, int] = tuple(new_indices[-2:])
                elif len(new_indices) > 1 and new_indices[-2] ==  new_index:
                    pair_to_update: Tuple[int, int] = tuple(new_indices[-2:])
                
                if len(pair_to_update) == 2:
                    counts_and_locations[pair_to_update] = counts_and_locations.get(pair_to_update, [0, []])
                    counts_and_locations[pair_to_update][0] += 1
                    counts_and_locations[pair_to_update][1].append(indices_index)
                    if pair_to_update not in pairs_to_update_counts:
                        pairs_to_update_counts.append(pair_to_update)
            
            if len(new_pair) == 2:
                new_pair.popleft()

        indices[indices_index] = new_indices.copy()

    return indices, counts_and_locations, pairs_to_update_counts, negatives


def train_BPE(input_path:str, vocab_size:int, special_tokens:List[str]) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
        regex:re.Pattern = get_regex_pattern("GPT4")
        pre_tokenized: List[str] = pre_tokenize(regex, input_path)

        counts_and_locations: Dict[Tuple[int,int], list] = {}
        heap: List[Tuple[int, Tuple[int, int]]] = []
        negatives: Set[Tuple[int, int]] = set()
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
            max_counts, pair = heapq.heappop(heap)

            while pair in negatives:
                max_counts, pair = heapq.heappop(heap)

            # ties: List[Tuple[int, int]] = [pair]
            # while heap[0][0] == max_counts:
            #     next_pair: Tuple[int, int] = heapq.heappop(heap)[1]
            #     ties.append(next_pair)

            # sorted_ties: List[Tuple[int, int]] = sorted(ties, reverse=True)
            # for i in range(len(sorted_ties)):     
            #     if i == 0:
            #         pair = sorted_ties[i][1]
            #     else:
            #         heapq.heappush(heap, (max_counts, sorted_ties[i][1]))

            new_index = start_index + idx
            merges.append((vocab[pair[0]], vocab[pair[1]]))
            vocab[new_index] = vocab[pair[0]] + vocab[pair[1]]

            indices, counts_and_locations, pairs_to_update_counts, negatives = merge(indices, pair, new_index, counts_and_locations, negatives)

            for key in pairs_to_update_counts:
                heapq.heappush(heap, (-counts_and_locations[key][0], key))

        save_merges_and_vocab(merges, vocab, prefix="test")
        
        return vocab, merges

vocab, merges = train_BPE("/home/dk/code/minbpe/tests/taylorswift.txt", 512, [])

