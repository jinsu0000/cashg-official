import torch
from torch.utils.data import Sampler
from typing import Iterator, List
import random
from collections import defaultdict


class UniqueWriterBatchSampler(Sampler):

    def __init__(self, dataset, batch_size: int, drop_last: bool = True, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle


        self.writer_to_indices = defaultdict(list)
        for idx, item in enumerate(dataset.index):
            writer_id = item[0]
            self.writer_to_indices[writer_id].append(idx)

        self.writers = list(self.writer_to_indices.keys())
        self.num_writers = len(self.writers)

        if self.num_writers < batch_size:
            raise ValueError(
                f"Number of unique writers ({self.num_writers}) is less than batch_size ({batch_size}). "
                f"Cannot create batches with unique writers."
            )

        print(f"[UniqueWriterBatchSampler] Initialized with {self.num_writers} unique writers, batch_size={batch_size}")

    def __iter__(self) -> Iterator[List[int]]:


        import torch
        import time

        while True:

            if self.shuffle:
                seed = int(time.time() * 1000000) % (2**32)
                random.seed(seed)


            if self.shuffle:
                writers_shuffled = random.sample(self.writers, len(self.writers))
            else:
                writers_shuffled = self.writers.copy()


            batches = []
            for i in range(0, len(writers_shuffled), self.batch_size):
                batch_writers = writers_shuffled[i:i + self.batch_size]

                if len(batch_writers) < self.batch_size:
                    if self.drop_last:
                        break


                batch_indices = []
                for writer in batch_writers:
                    writer_samples = self.writer_to_indices[writer]
                    selected_idx = random.choice(writer_samples)
                    batch_indices.append(selected_idx)

                batches.append(batch_indices)


            if self.shuffle:
                random.shuffle(batches)


            for batch in batches:
                yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.writers) // self.batch_size
        else:
            return (len(self.writers) + self.batch_size - 1) // self.batch_size


class UniqueWriterSampler(Sampler):

    def __init__(self, dataset, shuffle: bool = True):
        self.dataset = dataset
        self.shuffle = shuffle


        self.writer_to_indices = defaultdict(list)
        for idx, item in enumerate(dataset.index):
            writer_id = item[0]
            self.writer_to_indices[writer_id].append(idx)

        self.writers = list(self.writer_to_indices.keys())

    def __iter__(self) -> Iterator[int]:

        writer_order = self.writers.copy()
        if self.shuffle:
            random.shuffle(writer_order)

        for writer in writer_order:
            samples = self.writer_to_indices[writer]
            yield random.choice(samples)

    def __len__(self) -> int:
        return len(self.writers)
