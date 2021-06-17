import numpy as np
import torch

import config


def jaccard_array(a, b):
    """Calculates Jaccard on arrays."""
    a = set(a)
    b = set(b)
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def process_data(text, dataset_label,
                 tokenizer, max_len):
    """Preprocesses one data sample and returns a dict
    with targets and other useful info.
    """
    text = ' ' + ' '.join(str(text).split())
    dataset_label = ' ' + ' '.join(str(dataset_label).split())

    len_sel_text = len(dataset_label) - 1

    # Get sel_text start and end idx
    idx_0 = None
    idx_1 = None
    for ind in (i for i, e in enumerate(text) if e == dataset_label[1]):
        if ' ' + text[ind:ind + len_sel_text] == dataset_label:
            idx_0 = ind
            idx_1 = ind + len_sel_text - 1
            break

    # Assign 1 as target for each char in sel_text
    char_targets = [0] * len(text)
    if idx_0 is not None and idx_1 is not None:
        for ct in range(idx_0, idx_1 + 1):
            char_targets[ct] = 1

    tokenized_text = tokenizer.encode(text)
    # Vocab ids
    input_ids_original = tokenized_text.ids
    print(f'Len text(by word): {len(text.split())}')
    print(f'Len text(by char): {len(text)}')
    print(f'Len input_ids_original: {len(input_ids_original)}')
    # Start and end char
    text_offsets = tokenized_text.offsets

    # Get ids (of word) within text of words that have target char
    target_ids = []
    for i, (offset_0, offset_1) in enumerate(text_offsets):
        if sum(char_targets[offset_0:offset_1]) > 0:
            target_ids.append(i)

    targets_start = target_ids[0]
    targets_end = target_ids[-1]

    # Soft Jaccard labels
    # ----------------------------------
    n = len(input_ids_original)
    sentence = np.arange(n) 
    answer = sentence[targets_start:targets_end + 1]

    start_labels = np.zeros(n)
    for i in range(targets_end + 1):
        jac = jaccard_array(answer, sentence[i:targets_end + 1])
        start_labels[i] = jac + jac ** 2
    start_labels = (1 - config.SOFT_ALPHA) * start_labels / start_labels.sum()
    start_labels[targets_start] += config.SOFT_ALPHA

    end_labels = np.zeros(n)
    for i in range(targets_start, n):
        jac = jaccard_array(answer, sentence[targets_start:i + 1])
        end_labels[i] = jac + jac ** 2
    end_labels = (1 - config.SOFT_ALPHA) * end_labels / end_labels.sum()
    end_labels[targets_end] += config.SOFT_ALPHA

    start_labels = [0] + list(start_labels) + [0]
    end_labels = [0] + list(end_labels) + [0]
    # ----------------------------------

    # Input for RoBERTa
    input_ids = [0] + input_ids_original + [2]
    # No token types in RoBERTa
    token_type_ids = [0] + [0] * (len(input_ids_original) + 1)
    # Mask of input without padding
    mask = [1] * len(token_type_ids)
    # Start and end char ids for each word including new tokens
    text_offsets = [(0, 0)] + text_offsets + [(0, 0)]
    # Ids within text of words that have target char including new tokens
    targets_start += 1
    targets_end += 1
    orig_start = 1
    orig_end = len(input_ids_original) -1 + 1

    # Input padding: new mask, token type ids, text offsets
    padding_len = max_len - len(input_ids)
    if padding_len > 0:
        input_ids = input_ids + ([1] * padding_len)
        mask = mask + ([0] * padding_len)
        token_type_ids = token_type_ids + ([0] * padding_len)
        text_offsets = text_offsets + ([(0, 0)] * padding_len)
        start_labels = start_labels + ([0] * padding_len)
        end_labels = end_labels + ([0] * padding_len)

    targets_select = [0] * len(token_type_ids)
    for i in range(len(targets_select)):
        if i in target_ids:
            targets_select[i + 1] = 1

    return {'ids': input_ids,
            'mask': mask,
            'token_type_ids': token_type_ids,
            'start_labels': start_labels,
            'end_labels': end_labels,
            'orig_start': orig_start,
            'orig_end': orig_end,
            'orig_text': text,
            'orig_dataset_label': dataset_label,
            'offsets': text_offsets,
            'targets_select': targets_select}


class ColeridgeDataset:
    def __init__(self, texts, dataset_labels):
        self.texts = texts
        self.dataset_labels = dataset_labels
        self.max_len = config.MAX_LEN
        self.tokenizer = config.TOKENIZER

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        """Returns preprocessed data sample as dict with
        data converted to tensors.
        """
        data = process_data(self.texts[item],
                            self.dataset_labels[item],
                            self.tokenizer,
                            self.max_len)

        return {'ids': torch.tensor(data['ids'], dtype=torch.long),
                'mask': torch.tensor(data['mask'], dtype=torch.long),
                'token_type_ids': torch.tensor(data['token_type_ids'],
                                               dtype=torch.long),
                'start_labels': torch.tensor(data['start_labels'],
                                             dtype=torch.float),
                'end_labels': torch.tensor(data['end_labels'],
                                           dtype=torch.float),
                'orig_start': data['orig_start'],
                'orig_end': data['orig_end'],
                'orig_text': data['orig_text'],
                'orig_dataset_label': data['orig_dataset_label'],
                'offsets': torch.tensor(data['offsets'], dtype=torch.long),
                'targets_select': torch.tensor(data['targets_select'],
                                               dtype=torch.float)}
