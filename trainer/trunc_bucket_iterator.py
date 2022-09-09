import math
import torch
import random

class TruncBucketIterator(object):
    def __init__(self, data, batch_size, sort_key='tokens_index', trunc="both", trunc_length=512, shuffle=True, sort=True, device='cpu'):
        self.shuffle = shuffle
        self.sort = sort
        self.sort_key = sort_key
        self.device=device
        self.trunc_length=trunc_length
        self.trunc = trunc
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
        batch_labels = []
        batch_usr_indeces = []
        batch_prd_indeces = []

        max_len = max([len(t[self.sort_key]) for t in batch_data])
        if max_len > self.trunc_length: max_len: max_len = self.trunc_length
        for item in batch_data:
            tokens, tokens_index, label, usr_index, prd_index = item["tokens"], item["tokens_index"], item["label"], item["usr_index"], item["prd_index"]

            if len(tokens_index) > max_len:
                if self.trunc == 'head':
                    tokens_index = tokens_index[:self.trunc_length-1] + tokens_index[-1:]
                if self.trunc == 'tail':
                    tokens_index = tokens_index[0:1] +  tokens_index[-self.trunc_length-1:]
                if self.trunc == 'both':
                    tokens_index= tokens_index[:128] + tokens_index[-self.trunc_length + 128:]
            else:
                tokens_padding = [0] * (max_len - len(tokens_index))
                tokens_index = tokens_index + tokens_padding

            batch_usr_indeces.append(usr_index)
            batch_prd_indeces.append(prd_index)
            batch_text_indices.append(tokens_index)
            batch_labels.append(label)

        batch_text_indices = torch.tensor(batch_text_indices, device=self.device)
        batch_labels = torch.tensor(batch_labels, device=self.device)
        batch_usr_indeces=torch.tensor(batch_usr_indeces, device=self.device, dtype=torch.long)
        batch_prd_indeces=torch.tensor(batch_prd_indeces, device=self.device, dtype=torch.long)

        return {'batch_text_indices': batch_text_indices,
                'batch_labels': batch_labels,
                'batch_usr_indeces': batch_usr_indeces,
                'batch_prd_indeces': batch_prd_indeces
                }

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        for idx in range(self.batch_len):
            yield self.batches[idx]

    def __len__(self):
        return self.batch_len