import regex as re
from dataclasses import dataclass
from typing import List,Tuple, Dict, DefaultDict, OrderedDict as OD
from collections import defaultdict, OrderedDict
import heapq
import json

def get_regex_pattern() -> str: 
    return re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

def pre_tokenize(pattern:str, filename:str, endLine:bool=False) -> str:
    with open(filename, "r") as F:
        if not endLine:
            pre_tokenized_output:str = "".join(re.findall(pattern, "".join(line + " " for line in F.read().splitlines())))
        else:
            pre_tokenized_output:str = "".join(re.findall(pattern, F.read()))
    return pre_tokenized_output.strip()


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
        regex:re.Pattern = get_regex_pattern()
        pre_tokenized = pre_tokenize(regex, input_path)
        print(pre_tokenized)
        # with open(input_path, "r") as text:
            # pre_tokenized = F.read().encode("utf-8")
        pre_tokenized_bytes:bytes = pre_tokenized.encode("utf-8")
        del pre_tokenized
        vocab: Dict[int, bytes]  = {i:bytes(special_tokens[i].encode("utf-8")) for i in range(len(special_tokens))}

        location_indices: OD = OrderedDict()  #Ordered so we can use it as LRU cache when we hit vocab limit
        indices: list = []
        merges: List[Tuple[bytes, bytes]] = []
        for x in range(256):
            vocab[x + len(special_tokens)] = bytes([x])

        for i in range(len(pre_tokenized_bytes)):
            index = int(pre_tokenized_bytes[i])
            indices.append(index)
            not_new_slice: list = location_indices.get(index, [])
            if not_new_slice:
                location_indices[index].append(i)
            else:
                not_new_slice.append(i)
                location_indices[index] = not_new_slice

        start_index = len(vocab)
        counts: DefaultDict[Tuple[int, int], int] = defaultdict(int)
        heap: List[Tuple[int, Tuple[int, int]]] = []
        for pair in zip(indices, indices[1:]):
            counts[pair] += 1
        
        for key, value in counts.items():
            heapq.heappush(heap, (-value, key))
        

        curr_index = 0
        while heap and len(vocab) < vocab_size:
            #most common pair maybe need to add ordering logic in case of ties
            pair: Tuple[int, int]  = (0, 0)
            max_counts: int = 0

            max_counts, pair = heapq.heappop(heap)

            ties: List[Tuple[str, Tuple[int, int]]] = [("".join([vocab[x].decode("utf-8") for x in pair]), pair)]
            while heap[0][0] == max_counts:
                next_pair: Tuple[int, int] = heapq.heappop(heap)[1]
                ties.append(("".join([vocab[x].decode("utf-8") for x in next_pair]), next_pair))

            sorted_ties:List[Tuple[str, Tuple[int, int]]] = sorted(ties, reverse=True)
            for i in range(len(sorted_ties)):     
                if i == 0:
                    pair = sorted_ties[i][1]
                else:
                    heapq.heappush(heap, (max_counts, sorted_ties[i][1]))

            new_index = start_index + curr_index
            curr_index += 1
            merges.append((vocab[pair[0]], vocab[pair[1]]))
            vocab[new_index] = vocab[pair[0]] + vocab[pair[1]]

            indices = merge(indices, pair, new_index)


            # if len(vocab) > vocab_size:
            #     vocab.pop(location_indices.popitem(last=False)[0]) # LRU removal scheme
        # with open("merges.txt", "w") as F:
        #     for merge_item in merges:
        #         F.write(f"{merge_item[0]} {merge_item[1]}\n" )
        # jsondump = {}
        # for key, value in vocab.items():
        #     jsondump[value.decode("utf-8")] = key

        
        # with open("vocab.json", "w") as F:
        # #     json.dump(jsondump, F)
        # with open("my_merges2.txt", "w") as f:
        #     for key, val in merges:
        #         f.write(f"({key} {val})\n")
        
        # with open("my_vocab2.txt", "w") as f:
        #     for key, val in vocab.items():
        #         f.write(f"({key}:{val})\n")

        
        return vocab, merges
# class BPETokenizer():
#     def __init__(self, input_path:str, vocab_size:int, special_tokens:List[str]) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
#         pass

#     def encode(self, text: str) -> List[int]:
#         pre_tokenized = pre_tokenize(get_regex_pattern(), self.input_path)
#         # Note: this is a very slow implementation
#         for pair, new_index in self.params.merges.items():
#             indices = merge(indices, pair, new_index)
#         return indices

#     def decode(self, indices: List[int]) -> str:
#         bytes_list = list(map(self.vocab.get, indices))
#         text = b"".join(bytes_list).decode("utf-8")
#         return text


# pattern = get_regex_pattern()
# print(pre_tokenize(pattern, "test.txt"))

vocab, merges = train_BPE("tests/fixtures/corpus.en", 500, ['<|endoftext|>'])

