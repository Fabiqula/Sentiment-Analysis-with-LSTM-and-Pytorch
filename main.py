import torch
import torch.nn as nn
from torchtext.datasets import IMDB
from torch.utils.data import random_split
import re
from collections import Counter, OrderedDict

#  Create the datasets
train_dataset = IMDB(split='train', root='./')
test_dataset = IMDB(split='test', root='./')


torch.manual_seed(1)
train_dataset, valid_dataset = random_split(list(train_dataset),[20000, 5000])

# View the data
# for i, (label, text) in enumerate(train_dataset):
#     if i < 5:
#         print(f'{i}')
#         print(f'{label}')
#         print(f'{text}')
#     else:
#         break

# find unique tokens (words)

def tokenizer(text):
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Convert to lowercase
    text = text.lower()

    # Extract emoticons
    emoticons = re.findall(r'(:|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())

    # Replace non-word characters with spaces
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')


    # Tokenize text
    tokens = text.split()

    return tokens

token_counts = Counter()
for label, line in train_dataset:
    tokens = tokenizer(line)
    token_counts.update(tokens)
print('Vocab_size', len(token_counts))

from torchtext.vocab import vocab
sorted_by_freq_tuples = token_counts.most_common()

ordered_dict = OrderedDict(sorted_by_freq_tuples)
vocab = vocab(ordered_dict)
vocab.insert_token("<pad>", 0)
vocab.insert_token("<unk>", 1)
vocab.set_default_index(1)

print([vocab[token] for token in ['this', 'is', 'an', 'example']])

# define the functions for transformation

text_pipeline = lambda x: [vocab[token] for token in tokenizer(x)]
label_pipeline = lambda x: 1. if x =='pos' else 0.

# wrap the encode and transformation function

def collate_batch(batch):
    label_list, text_list, lengths = [], [], []
    for _label, _text in batch:
        label_list.append(label_pipeline(_label))embed
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        lengths.append(processed_text.shape[0])
    label_list = torch.tensor(label_list, dtype=torch.int64)
    lengths = torch.tensor(lengths, dtype=torch.int64)
    padded_text_list = nn.utils.rnn.pad_sequence(
        text_list, batch_first=True])
    return padded_text_list, label_list, lengths

from torch.utils.data import DataLoader

data_loader = DataLoader(train_dataset, batch_size=4, shuffle=False, collate_fn=collate_batch)