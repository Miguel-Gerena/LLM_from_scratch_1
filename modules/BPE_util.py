from copy import deepcopy
import regex as re
from dataclasses import dataclass
from typing import List,Tuple, Dict
from collections import defaultdict
import joblib
from tqdm import tqdm
import multiprocessing


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

def generate_sentence_buckets(regex:re.Pattern, filename:str, EOF_token:str='<|endoftext|>', num_proc:int=6) -> List[str]:
    sentence_buckets = ["" for _ in range(num_proc)]
    i:int = 0
    bucket:str = ""
    chunk:int = 0
    p = multiprocessing.Pool(num_proc)
    processes: list =[]
    with open(filename, "r") as F:
        total_num_lines = len(F.readlines())
        F.seek(0,0)
        pbar = tqdm(F, desc=f"Distributing file into buckets", total=total_num_lines)
        for line in pbar:
            bucket += line
            if (line[-1] == "\n" or line == EOF_token or line == EOF_token+"\n") and chunk >= total_num_lines//num_proc:
                processes.append(p.apply_async(re.findall, (regex, bucket)))
                i += 1
                chunk = 0
                bucket = ""
                pbar.set_description(f"launched proceess {i} of {num_proc}")
            chunk += 1
        if bucket != "":  # catch the last process if chunks are skewed
            processes.append(p.apply_async(re.findall, (regex, bucket)))

    print("Tokenizing and waiting for other processes")
    for i in range(len(sentence_buckets)):
        sentence_buckets[i] = processes[i].get()
        assert len(sentence_buckets[i])> 1, f"Bucket {i} is empty"   
    return sentence_buckets

def generate_pairs_and_location_from_buckets(sentence_bucket:List[str]) -> Tuple[List[List[int]], Dict[Tuple[int, int], list], int]:
    indices: List[List[int]] = []
    counts_and_locations: Dict[Tuple[int, int], list] = {}
    idx:int=0
    for sentence in [sentence_bucket]:
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

    # info = ""
    # total = 0
    # for a in ans:
    #     pair = max(a[1], key=lambda x: a[1][x][0])
    #     info += f"pair: {max(a[1], key=lambda x: a[1][x][0])}, counts: {a[1][pair][0]} \n"
    #     if pair == (32, 116):
    #         total += a[1][pair][0]


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
            for index, value in locations.items():
                final_counts_locations[pair][1][index + last_offset] += value
        last_offset = offset
    return final_indices, final_counts_locations
    
def build_pairs_and_locations(indices: List[int], counts_and_locations: Dict[Tuple[int, int], list], index: int) -> Dict[Tuple[int, int], list]:
    for pair in zip(indices, indices[1:]):
        counts_and_locations[pair] = counts_and_locations.get(pair, [0, defaultdict(int)])
        counts_and_locations[pair][0] += 1
        counts_and_locations[pair][1][index] += 1
    return counts_and_locations