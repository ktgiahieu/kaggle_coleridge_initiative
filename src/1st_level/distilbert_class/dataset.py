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
	encoded_dict = tokenizer.encode_plus(
                        text,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = config.MAX_LEN,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                        truncation = False,
                   )
    # Vocab ids
    input_ids_original = tokenized_text.ids

    # ----------------------------------

    # Input for BERT
    input_ids = encoded_dict['input_ids']
    # No token types in BERT
    token_type_ids = encoded_dict['token_type_ids ']
    # Mask of input without padding
    mask = encoded_dict['attention_mask']

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
