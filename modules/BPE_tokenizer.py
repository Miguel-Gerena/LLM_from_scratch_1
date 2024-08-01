import multiprocessing
import regex as re
from typing import List,Tuple, Dict, Deque, Set
from collections import defaultdict, deque
import heapq
import time
import resource
import os
import pickle
import gc
from tqdm import tqdm

from BPE_util import (BPETokenizerParams, build_pairs_and_locations,
                    generate_sentence_buckets, pre_tokenize, gpt2_bytes_to_unicode,
                    reduce_pairs_and_locations_from_buckets, get_regex_pattern) 


class BPE():
    def __init__(self) -> None:
        self.params = BPETokenizerParams({}, {})
        self.special_tokens: Set[str] = set()

    def _get_pairs_and_locations(self, regex:re.Pattern , filename:str) ->  Tuple[List[List[int]], Dict[Tuple[int, int], list]]:
        indices: List[List[int]] = []
        counts_and_locations: Dict[Tuple[int, int], list] = {}
        idx:int = 0
        for sentence in tqdm(pre_tokenize(regex, filename), total=lines):
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
        print("Training")
        
        if useHeap:
            heap: List[Tuple[int, Tuple[int, int]]] = []
            for key, value in counts_and_locations.items():
                heapq.heappush(heap, (-value[0], key))

        start_index = len(self.params.vocab)
        num_merges = vocab_size - start_index
        gc.collect()
        for idx in tqdm(range(num_merges)):
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
    def __init__(self, num_procs:int = 6, EOF_token:str="<|endoftext|>",  *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_procs:int = multiprocessing.cpu_count() if num_procs == -1 else num_procs
        self.EOF_token = EOF_token

    def _get_pairs_and_locations(self, regex:re.Pattern , filename:str) ->  Tuple[List[List[int]], Dict[Tuple[int, int], list]]:
        tokenized_buckets = generate_sentence_buckets(regex, filename, EOF_token=self.EOF_token, num_proc=self.num_procs)
        return reduce_pairs_and_locations_from_buckets(tokenized_buckets, self.num_procs)


# vocab, merges = train_BPE("/home/dk/code/minbpe/tests/taylorswift.txt", 512, [])

with open(r"data/TinyStoriesV2-GPT4-train.txt", 'r') as fp:
    lines = len(fp.readlines())
    print('Total Number of lines:', lines)


# t0 = time.time()
# bpe = BPE()
# bpe.train("data/TinyStoriesV2-GPT4-train.txt", 10000, ["<|endoftext|>"], "GPT2")
# t1 = time.time()
# print(f"Training took {t1 - t0:.2f} seconds")
# bpe.serialize_merges_and_vocab("tinyStories_train")
# print(f"memory usage: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**3} Gb")

if __name__ == "__main__":
    t0 = time.time()
    bpe = BPE()
    bpe.train("data/TinyStoriesV2-GPT4-train.txt", 10000, ["<|endoftext|>"], "GPT2")
    t1 = time.time()
    print(f"Training took {t1 - t0:.2f} seconds")
    # bpe.serialize_merges_and_vocab("parallel__val")
    bpe.save_merges_and_vocab_to_txt("parallel")
    print(f"memory usage: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**3:.4f} Gb")


