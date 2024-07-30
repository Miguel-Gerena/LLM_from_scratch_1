from copy import deepcopy
import regex as re
from dataclasses import dataclass
from typing import List,Tuple, Dict, DefaultDict, Deque, Set, OrderedDict as OD
from collections import defaultdict, OrderedDict, deque
import heapq
import time
import resource
import os
import pickle
import joblib
from tqdm import tqdm


@dataclass(frozen=True)
class BPETokenizerParams:
    vocab: Dict[int, bytes]            
    merges: Dict[Tuple[int, int], int] 

@dataclass(frozen=True)
class BPETokenizerParamsBytes:
    vocab: Dict[int, bytes]            
    merges: List[Tuple[bytes, bytes]] 

def gpt2_bytes_to_unicode(offset:int) -> dict[int, bytes]:
    """
    Returns a mapping between every possible byte (an integer from 0 to 255) to a
    printable unicode string character representation. This function is taken
    from the GPT-2 code."""
    bs = (
    list(range(ord(" "), ord("~") + 1))
    + list(range(ord("¡"), ord("¬") + 1))
    + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    # now get the representations of the other 68 integers that do need shifting
    # each will get mapped chr(256 + n), where n will grow from 0...67 in the loop
    # Get printable representations of the remaining integers 68 integers.
    n = 0
    for b in range(offset, 2**8 + offset):
        if b not in bs:
            # If this integer isn't in our list of visually-representable
            # charcters, then map it to the next nice character (offset by 256)
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n).encode("utf-8") for n in cs]
    d = dict(zip(bs, characters))
    return d

def get_regex_pattern(pattern:str) -> re.Pattern[str]: 
    reg: str = ""
    if pattern == "GPT2":
        reg = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    elif pattern == "GPT4":
        reg = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
    elif pattern == "BPE_Example":
        reg = r"""\S"""
    else:
        reg = fr"""{pattern}"""
    return re.compile(reg)

def pre_tokenize(pattern:re.Pattern , filename:str) -> List[str]:
    with open(filename, "r") as F:
        for line in F:
            yield re.findall(pattern, line)

def generate_sentence_buckets(filename:str, EOF_token:str='<|endoftext|>', num_proc:int=6) -> List[str]:
    sentence_buckets = ["" for _ in range(num_proc)]
    i = 0
    with open(filename, "r") as F:
        for line in F:
            sentence_buckets[i%num_proc] += line
            if line[-1] == EOF_token or line == EOF_token or line == EOF_token+"\n":
                i += 1
    return sentence_buckets

def pre_tokenize_buckets(pattern:re.Pattern, sentences_buckets:List[str], num_procs:int=6) -> List[str]:
    ans = joblib.Parallel(n_jobs=num_procs, backend="loky")(
    joblib.delayed(re.findall)(pattern, sentence_bucket)
    for sentence_bucket in tqdm(sentences_buckets, total=len(sentences_buckets), desc="Tokenizing Buckets"))
    return ans

def generate_pairs_and_location_from_buckets(sentence_bucket:List[str]) -> Tuple[List[List[int]], Dict[Tuple[int, int], list], int]:
    indices: List[List[int]] = []
    counts_and_locations: Dict[Tuple[int, int], list] = {}
    idx:int=0
    for sentence in sentence_bucket:
        for word in sentence:
            word_ints: List[int] = list(word.encode("utf-8"))
            indices.append(word_ints)
            counts_and_locations = build_pairs_and_locations(word_ints, counts_and_locations, idx)
            idx += 1
    return indices, counts_and_locations, idx

def reduce_pairs_and_locations_from_buckets(sentence_buckets:List[str], num_procs:int=6) -> Tuple[List[List[int]], Dict[Tuple[int, int], list]]:
    ans = joblib.Parallel(n_jobs=num_procs, backend="loky")(
    joblib.delayed(generate_pairs_and_location_from_buckets)(sentence_bucket)
    for sentence_bucket in tqdm(sentence_buckets, total=len(sentence_buckets)))

    final_indices:list[List[int]] = []
    final_counts_locations = {}
    last_offset:int = 0
    for indices, counts_and_location_shard, offset in tqdm(ans, desc="Reducing parallel output"):
        assert offset == len(indices), f"offset: {offset} is wrong. len: {len(indices)}"
        if not final_indices:
            final_indices = deepcopy(indices)
            final_counts_locations = deepcopy(counts_and_location_shard)
            last_offset = offset
            continue

        final_indices.extend(indices)
        for pair, (counts, locations) in counts_and_location_shard.items():
            final_counts_locations[pair] = final_counts_locations.get(pair, [0, defaultdict(int)])
            final_counts_locations[pair][0] += counts
            for index in locations:
                final_counts_locations[pair][1][index + last_offset] += 1
        last_offset = offset
    return final_indices, final_counts_locations
    
def build_pairs_and_locations(indices: List[int], counts_and_locations: Dict[Tuple[int, int], list], index: int) -> Dict[Tuple[int, int], list]:
    for pair in zip(indices, indices[1:]):
        counts_and_locations[pair] = counts_and_locations.get(pair, [0, defaultdict(int)])
        counts_and_locations[pair][0] += 1
        counts_and_locations[pair][1][index] += 1
    return counts_and_locations

class BPE():
    def __init__(self) -> None:
        self.params = BPETokenizerParams({}, {})
        self.special_tokens: Set[str] = set()

    def _get_pairs_and_locations(self, regex:re.Pattern , filename:str) ->  Tuple[List[List[int]], Dict[Tuple[int, int], list]]:
        indices: List[List[int]] = []
        counts_and_locations: Dict[Tuple[int, int], list] = {}
        idx:int = 0
        for sentence in pre_tokenize(regex, filename):
            for word in sentence:
                word_ints: List[int] = list(word.encode("utf-8"))
                indices.append(word_ints)
                counts_and_locations = build_pairs_and_locations(word_ints, counts_and_locations, idx)
                idx += 1
        return indices, counts_and_locations

    def save_merges_and_vocab_to_txt(self, prefix="") -> None:
        with open(f"{prefix}_merges.txt", "w") as f:
            for key, val in self.params.merges:
                f.write(f"{key}:{val}\n")
        
        with open(f"{prefix}_vocab.txt", "w") as f:
            for key, val in self.params.vocab.items():
                f.write(f"{key}:{val}\n")
    
    def serialize_merges_and_vocab(self, prefix="") -> None:
        with open(prefix + "_merges.pkl", "wb") as F:
            pickle.dump(self.params.merges, F)
        with open(prefix + "_vocab.pkl", "wb") as F:
            pickle.dump(self.params.vocab, F)
    
    def load_serialized_merges_and_vocab(self, prefix="") -> None:
        with open(prefix + "_merges.pkl", "rb") as F:
            merges = pickle.load(F)
        with open(prefix + "_vocab.pkl", "rb") as F:
            vocab = pickle.load(F)
        self.params = BPETokenizerParams(vocab, merges)
    
    def _merge(self, indices: List[List[int]], pair: Tuple[(int, int)], new_index: int, counts_and_locations: Dict[Tuple[int, int], list]) -> Tuple[List[List[int]],  Dict[Tuple[int, int], list],  List[Tuple[int, int]]]:
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
                    # token has changed.  update the count of surrounding pairs
                    if len(new_indices) > 2 and counts_and_locations.get((new_indices[-3], new_indices[-2]), []):  
                        prev_pair = (new_indices[-3], new_indices[-2])
                        counts_and_locations[prev_pair][0] -= 1
                        counts_and_locations[prev_pair][1][indices_index] -= 1
                        if counts_and_locations[prev_pair][1][indices_index] <= 0:
                            counts_and_locations[prev_pair][1].pop(indices_index)

                    if word_ints and counts_and_locations.get((new_indices[-1], word_ints[0]), []):
                        next_pair = (new_indices[-1], word_ints[0])
                        counts_and_locations[next_pair][0] -= 1
                        counts_and_locations[next_pair][1][indices_index] -= 1
                        if counts_and_locations[next_pair][1][indices_index] <= 0:
                            counts_and_locations[next_pair][1].pop(indices_index)

                    new_indices.pop()
                    new_indices.pop()
                    #  add new token for the given pair
                    new_indices.append(new_index)
                    new_pair = deque([new_index])

                if new_indices:
                    pair_to_update: Tuple[int, int] = (-1, -1)
                    if new_indices[-1] == new_index:
                        pair_to_update: Tuple[int, int] = tuple(new_indices[-2:])
                    elif len(new_indices) > 1 and new_indices[-2] ==  new_index:
                        pair_to_update: Tuple[int, int] = tuple(new_indices[-2:])
                    
                    if len(pair_to_update) == 2 and pair_to_update != (-1, -1):
                        counts_and_locations[pair_to_update] = counts_and_locations.get(pair_to_update, [0, defaultdict(int)])
                        counts_and_locations[pair_to_update][0] += 1
                        counts_and_locations[pair_to_update][1][indices_index] += 1
                        if pair_to_update not in pairs_to_update_counts:
                            pairs_to_update_counts.append(pair_to_update)
                
                if len(new_pair) == 2:
                    new_pair.popleft()

            indices[indices_index] = new_indices
        return indices, counts_and_locations, pairs_to_update_counts
    
    def train(self, input_path:str | os.PathLike, vocab_size:int, special_tokens:List[str], regex_pattern:str="GPT4", debug:bool=False, useControl_characters:bool = False, useHeap: bool = False) -> None:
        regex:re.Pattern = get_regex_pattern(regex_pattern)      
        
        for i in range(len(special_tokens)):
            self.special_tokens.add(special_tokens[i])
            self.params.vocab[len(self.params.vocab)] = bytes(special_tokens[i].encode("utf-8"))

        if useControl_characters:
            for x in range(256):
                self.params.vocab[x + len(special_tokens)] = bytes([x])
        else:
            self.params.vocab.update(gpt2_bytes_to_unicode(len(special_tokens)))

        indices, counts_and_locations = self._get_pairs_and_locations(regex, input_path)
        
        if useHeap:
            heap: List[Tuple[int, Tuple[int, int]]] = []
            for key, value in counts_and_locations.items():
                heapq.heappush(heap, (-value[0], key))

        start_index = len(self.params.vocab)
        num_merges = vocab_size - start_index
        for idx in range(num_merges):
            pair: Tuple[int, int]  = (0, 0)
            if useHeap:
                max_counts, pair = heapq.heappop(heap)
            else:
                pair = max(counts_and_locations, key=lambda x: counts_and_locations[x][0])
                max_counts, locations = counts_and_locations.pop(pair)

            if useHeap:
                # if counts in heap are stale update count for the current pair and retry heap
                while -max_counts != counts_and_locations[pair][0]:
                    if counts_and_locations[pair][0] <= 0:
                        max_counts, pair = heapq.heappop(heap)
                    else:
                        max_counts, pair = heapq.heappushpop(heap, (-counts_and_locations[pair][0], pair))

                ties = {pair:max_counts}
                while max_counts == heap[0][0]:
                    max_counts, pair = heapq.heappop(heap)
                    ties[pair] = max_counts
            else:
                ties = {pair:[max_counts, locations]}
                while next_pair := max(counts_and_locations, key=lambda x: counts_and_locations[x][0]):
                    if counts_and_locations[next_pair][0] != max_counts:
                        break
                    pair = next_pair
                    max_counts, locations = counts_and_locations.pop(pair)
                    ties[pair] = [max_counts, locations]

            if not useHeap:
                for key, value in ties.items():
                    counts_and_locations[key] = value

            if len(ties) > 1:
                sorted_ties = sorted([((self.params.vocab[char[0]] + self.params.vocab[char[1]]).decode("utf-8"), char, value) for char, value in ties.items()])
                if debug:
                    assert [p[0] for p in sorted_ties] == sorted([p[0] for p in ties]), f"sorted ties aren't sorted correctly"
                pair = sorted_ties[0][1]
                if useHeap:
                    for _, pair_to_heap, counts in sorted_ties:
                        if pair != pair_to_heap:
                            heapq.heappush(heap, (counts, pair_to_heap))
                del sorted_ties, ties       

            new_index = start_index + idx
            if new_index == 274:
                print()
            self.params.merges[(pair[0], pair[1])] = new_index
            self.params.vocab[new_index] = self.params.vocab[pair[0]] + self.params.vocab[pair[1]]

            if debug:
                print(f"merge {idx+1}/{num_merges}: {pair} -> {new_index} ({self.params.vocab[new_index]}) had {-max_counts} occurrences")

            indices, counts_and_locations, pairs_to_update_counts = self._merge(indices, pair, new_index, counts_and_locations)

            if useHeap:
                for key in pairs_to_update_counts:
                    heapq.heappush(heap, (-counts_and_locations[key][0], key))
    
    def decode(self, indices):
        sentence = []
        for idx in indices:
            if idx in self.params.vocab:
                sentence.append[self.params.vocab[idx]]
            else:
                raise ValueError(f"Invalid token id: {idx} resulting in unknown token")
        return b"".join(sentence).decode("utf-8", errors="replace")
    
    def encode(self, indices):
        sentence = []
        for idx in indices:
            if idx in self.params.vocab:
                sentence.append[self.params.vocab[idx]]
            else:
                raise ValueError(f"Invalid token id: {idx} resulting in unknown token")
        return b"".join(sentence).decode("utf-8", errors="replace")

class BPE_parallel(BPE):
    def __init__(self, num_procs:int = 6, EOF_token:str="\n",  *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_procs:int = num_procs
        self.EOF_token = EOF_token

    def _get_pairs_and_locations(self, regex:re.Pattern , filename:str) ->  Tuple[List[List[int]], Dict[Tuple[int, int], list]]:
        sentence_buckets = generate_sentence_buckets(filename, EOF_token=self.EOF_token, num_proc=self.num_procs)
        for i in range(len(sentence_buckets)):
            assert len(sentence_buckets[i]) > 1, f"empty bucket {i}"
        tokenized_buckets = pre_tokenize_buckets(regex, sentence_buckets, self.num_procs)
        return reduce_pairs_and_locations_from_buckets(tokenized_buckets, self.num_procs)


# vocab, merges = train_BPE("/home/dk/code/minbpe/tests/taylorswift.txt", 512, [])

# with open(r"data/TinyStoriesV2-GPT4-train.txt", 'r') as fp:
#     lines = len(fp.readlines())
#     print('Total Number of lines:', lines)


# t0 = time.time()
# bpe = BPE()
# bpe.train("data/TinyStoriesV2-GPT4-train.txt", 10000, ["<|endoftext|>"], "GPT2")
# t1 = time.time()
# print(f"Training took {t1 - t0:.2f} seconds")
# bpe.serialize_merges_and_vocab("tinyStories_train")
# print(f"memory usage: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**3} Gb")

t0 = time.time()
bpe = BPE_parallel()
bpe.train("tests/fixtures/corpus.en", 500, ["<|endoftext|>"], "GPT2")
t1 = time.time()
print(f"Training took {t1 - t0:.2f} seconds")
bpe.serialize_merges_and_vocab("parallel_tiny_val")
bpe.save_merges_and_vocab_to_txt("parallel")
print(f"memory usage: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**3} Gb")


