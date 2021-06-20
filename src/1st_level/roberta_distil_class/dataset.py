import numpy as np
import torch

import config


def jaccard_array(a, b):
    """Calculates Jaccard on arrays."""
    a = set(a)
    b = set(b)
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def process_data(text, label,
                 tokenizer, max_len):
    """Preprocesses one data sample and returns a dict
    with targets and other useful info.
    """
    text = ' ' + ' '.join(str(text).split())

    tokenized_text = tokenizer.encode(text)
    # Vocab ids
    input_ids_original = tokenized_text.ids

    # ----------------------------------

    # Input for RoBERTa
    input_ids = [0] + input_ids_original + [2]
    # No token types in RoBERTa
    token_type_ids = [0] + [0] * (len(input_ids_original) + 1)
    # Mask of input without padding
    mask = [1] * len(token_type_ids)

    # Input padding: new mask, token type ids, text offsets
    padding_len = max_len - len(input_ids)
    if padding_len > 0:
        input_ids = input_ids + ([1] * padding_len)
        mask = mask + ([0] * padding_len)
        token_type_ids = token_type_ids + ([0] * padding_len)

    return {'ids': input_ids,
            'mask': mask,
            'token_type_ids': token_type_ids,
            'labels': [label]}


class ColeridgeDataset:
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        self.max_len = config.MAX_LEN
        self.tokenizer = config.TOKENIZER

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        """Returns preprocessed data sample as dict with
        data converted to tensors.
        """
        data = process_data(self.texts[item],
                            self.labels[item],
                            self.tokenizer,
                            self.max_len)

        return {'ids': torch.tensor(data['ids'], dtype=torch.long),
                'mask': torch.tensor(data['mask'], dtype=torch.long),
                'token_type_ids': torch.tensor(data['token_type_ids'],
                                               dtype=torch.long),
                'labels': torch.tensor(data['labels'], dtype=torch.float),}
