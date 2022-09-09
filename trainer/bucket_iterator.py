import math
import torch
import random

class BucketIterator(object):
    def __init__(self, data, batch_size, sort_key='tokens_index', shuffle=True, sort=True, device='cpu'):
        self.shuffle = shuffle
        self.sort = sort
        self.sort_key = sort_key
        self.device=device
        self.batches = self.sort_and_pad(data, batch_size)
        self.batch_len = len(self.batches)

    def sort_and_pad(self, data, batch_size):
        num_batch = int(math.ceil(len(data) / batch_size))
        if self.sort:
            sorted_data = sorted(data, key=lambda x: len(x[self.sort_key]))
        else:
            sorted_data = data
        batches = []
        for i in range(num_batch):
            batches.append(self.pad_data(sorted_data[i*batch_size : (i+1)*batch_size]))
        return batches

    def pad_data(self, batch_data):
        batch_text_indices = []
        batch_aspect_indices = []
        batch_labels = []
        batch_aspect_masks = []

        max_len = max([len(t[self.sort_key]) for t in batch_data])
        max_aspect_len = max([len(t["aspect_tokens_index"]) for t in batch_data])
        for item in batch_data:
            tokens_index, aspect_tokens_index, aspect_mask, label = \
            item["tokens_index"], item["aspect_tokens_index"], item["aspect_mask"], item["label"]

            tokens_padding = [0] * (max_len - len(tokens_index))
            aspect_tokens_padding = [0] * (max_aspect_len - len(aspect_tokens_index))
            aspect_mask_padding = [0] * (max_len - len(aspect_mask))

            tokens_index.extend(tokens_padding)
            aspect_tokens_index.extend(aspect_tokens_padding)
            aspect_mask.extend(aspect_mask_padding)

            batch_text_indices.append(tokens_index)
            batch_aspect_indices.append(aspect_tokens_index)
            batch_labels.append(label)
            batch_aspect_masks.append(aspect_mask)

        batch_text_indices = torch.tensor(batch_text_indices, device=self.device)
        batch_aspect_indices = torch.tensor(batch_aspect_indices, device=self.device)
        batch_labels = torch.tensor(batch_labels, device=self.device)
        batch_aspect_masks = torch.tensor(batch_aspect_masks, device=self.device)

        return {'batch_text_indices': batch_text_indices,
                'batch_aspect_indices': batch_aspect_indices,
                'batch_labels': batch_labels,
                'batch_aspect_masks': batch_aspect_masks}

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        for idx in range(self.batch_len):
            yield self.batches[idx]

    def __len__(self):
        return self.batch_len