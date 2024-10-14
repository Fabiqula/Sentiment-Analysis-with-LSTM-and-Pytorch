import re
from collections import Counter, OrderedDict

import torch
import torch.nn as nn
from torchtext.datasets import IMDB

from torch.utils.data import DataLoader, random_split


def main():

    from torchtext.vocab import vocab

    #  Create the datasets
    train_dataset = IMDB(split='train', root='./')
    test_dataset = IMDB(split='test', root='./')

    torch.manual_seed(1)
    train_dataset, valid_dataset = random_split(list(train_dataset),[20000, 5000])

    #View the data
    # for i, (label, text) in enumerate(train_dataset):
    #     if i < 5:
    #         print(f'{i}')
    #         print(f'{label}')
    #         print(f'{text}')
    #     else:
    #         break

    #find unique tokens (words)

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


    sorted_by_freq_tuples = token_counts.most_common()

    ordered_dict = OrderedDict(sorted_by_freq_tuples)
    vocab = vocab(ordered_dict)
    vocab.insert_token("<pad>", 0)
    vocab.insert_token("<unk>", 1)
    vocab.set_default_index(1)

    # print([vocab[token] for token in ['this', 'is', 'an', 'example']])

    # define the functions for transformation

    text_pipeline = lambda x: [vocab[token] for token in tokenizer(x)]
    label_pipeline = lambda x: 1. if x == 1 else 0.


    # wrap the encode and transformation function

    def collate_batch(batch):
        label_list, text_list, lengths = [], [], []
        for _label, _text in batch:
            label_list.append(label_pipeline(_label))
            processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
            text_list.append(processed_text)
            lengths.append(processed_text.shape[0])
        label_list = torch.tensor(label_list, dtype=torch.float32)
        lengths = torch.tensor(lengths, dtype=torch.int64)
        padded_text_list = nn.utils.rnn.pad_sequence(
            text_list, batch_first=True)
        return padded_text_list, label_list, lengths


    #checking dataloder
    # XdataloaderX = DataLoader(train_dataset, batch_size=4, shuffle=False, collate_fn=collate_batch)
    # for text_batch, label_batch, lengths in XdataloaderX:
    #     print(f'Raw Labels: {label_batch}')
    #     break

    # text_batch, label_batch, length_batch = next(iter(XdataloaderX))
    # print(text_batch)

    batch_size = 32

    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    valid_dl = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    class RNN(nn.Module):
        def __init__(self, vocab_size, num_embd, rnn_hidden, fc_hidden):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, num_embd, padding_idx=0)
            self.rnn = nn.LSTM(num_embd, rnn_hidden, batch_first=True)

            self.fc1 = nn.Linear(rnn_hidden, fc_hidden)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(fc_hidden, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, text, lengths):
            out = self.embedding(text)
            out = nn.utils.rnn.pack_padded_sequence(out, lengths.cpu().numpy(), enforce_sorted=False, batch_first=True)
            out, (hidden, cell) = self.rnn(out)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
            out = hidden[-1, :, :]
            out = self.fc1(out)
            out = self.relu(out)
            out = self.fc2(out)
            out = self.sigmoid(out)
            return out


    vocab_size = len(vocab)
    num_embd = 20
    rnn_hidden = 64
    fc_hidden = 64
    torch.manual_seed(1)

    model = RNN(vocab_size, num_embd, rnn_hidden, fc_hidden)
    print(model)

    def train(dataloader):
        model.train()
        total_acc, total_loss = 0, 0
        for text_batch, label_batch, lengths in dataloader:
            optimizer.zero_grad()
            pred = model(text_batch, lengths)[:, 0]

            loss = loss_fn(pred, label_batch)
            loss.backward()
            optimizer.step()

            total_acc += ((pred > 0.5).float() == label_batch).float().sum().item()
            total_loss += loss.item() * label_batch.size(0)

        return total_acc / len(dataloader.dataset), total_loss / len(dataloader.dataset)

    def evaluate(dataloader):
        model.eval()
        total_acc, total_loss = 0, 0
        with torch.no_grad():
            for text_batch, label_batch, lengths in dataloader:

                pred = model(text_batch, lengths)[:, 0]
                loss = loss_fn(pred, label_batch)

                total_acc += ((pred > 0.5).float() == label_batch).float().sum().item()
                total_loss += loss.item() * label_batch.size(0)

            return total_acc / len(dataloader.dataset), total_loss / len(dataloader.dataset)

    loss_fn = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    num_epochs = 10
    torch.manual_seed(1)
    for epoch in range(num_epochs):
        acc_train, loss_train = train(train_dl)
        acc_valid, loss_valid = evaluate(valid_dl)
        print(f' Epoch {epoch}, accuracy: {acc_train:.4f}, val_accuracy: {acc_valid:.4f}')

if __name__ == "__main__":
    main()
