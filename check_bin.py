import numpy as np
import os
import random
import argparse

from crystallm import (
    CIFTokenizer,
)

def read_random_chunk(file_path, chunk_size):

    tokenizer = CIFTokenizer()

    file_size = os.path.getsize(file_path)

    mm = np.memmap(file_path, dtype='uint16', mode='r')

    start_pos = random.randint(0, file_size - chunk_size)

    chunk = mm[start_pos:start_pos+chunk_size]

    chunk_decode = tokenizer.decode(chunk)

    print(chunk)
    print(chunk_decode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Look at random chunk in .bin file")
    parser.add_argument("--file_path", type=str)
    parser.add_argument("--chunk_size",  type=int)
    args = parser.parse_args()

read_random_chunk(args.file_path, args.chunk_size)
    
