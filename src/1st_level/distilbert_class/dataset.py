import numpy as np
import torch

import config


def jaccard_array(a, b):
    """Calculates Jaccard on arrays."""
    a = set(a)
    b = set(b)
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

def process_all_data(texts, labels,
                 tokenizer, max_len):
    encoded_dict = tokenizer.batch_encode_plus(
        texts.tolist(),                      # Sentence to encode.
        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
        max_length = config.MAX_LEN,           # Pad & truncate all sentences.
        pad_to_max_length = True,
        return_attention_mask = True,   # Construct attn. masks.
        return_tensors = 'pt',     # Return pytorch tensors.
        truncation = False,
    )

    # ----------------------------------

    # Input for BERT
    input_ids = encoded_dict['input_ids']

    # Mask of input without padding
    mask = encoded_dict['attention_mask']
    return {'ids': input_ids,
            'mask': mask,
            'labels': labels}


class ColeridgeDataset:
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        self.max_len = config.MAX_LEN
        self.tokenizer = config.TOKENIZER
        self.data = process_all_data(self.texts, self.labels,
                 self.tokenizer, self.max_len)
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):

        return {'ids': torch.tensor(self.data['ids'][item], dtype=torch.long),
                'mask': torch.tensor(data['mask'][item], dtype=torch.long),
                'labels': torch.tensor(data['labels'][item], dtype=torch.float),}
