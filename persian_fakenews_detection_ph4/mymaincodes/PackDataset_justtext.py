import torch
from torch.utils.data import Dataset, DataLoader
# from torchtext import vocab as Vocab
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer, RobertaTokenizer, AutoTokenizer


bert_type = "HooshvareLab/bert-base-parsbert-uncased"
tokenizer = AutoTokenizer.from_pretrained(bert_type)


class processed_dataset_bert(Dataset):
    def __init__(self, data):
        self.texts = []
        self.labels = []
        for text, label in data:
            self.texts.append(torch.tensor(tokenizer.encode(text, max_length=128, truncation=True)))
            self.labels.append(label)
        assert len(self.texts) == len(self.labels)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


class packDataset_util_bert:
    def __init__(self):
        pass

    def fn(self, data):
        texts = []
        labels = []
        for text, label in data:
            texts.append(text)
            labels.append(label)
        labels = torch.tensor(labels)
        padded_texts = pad_sequence(texts, batch_first=True, padding_value=0)
        attention_masks = torch.zeros_like(padded_texts).masked_fill(padded_texts != 0, 1)
        return padded_texts, attention_masks, labels

    def get_loader(self, data, shuffle=True, batch_size=32):
        dataset = processed_dataset_bert(data)
        loader = DataLoader(dataset=dataset, shuffle=shuffle, batch_size=batch_size, collate_fn=self.fn)
        return loader


class processed_dataset_bert_with_category(Dataset):
    def __init__(self, data):
        # if bert_type == 'bert':
        #     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # else:
        #     tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

        self.texts = []
        self.categories = []
        self.labels = []
        for text, label, category in data:
            self.texts.append(torch.tensor(tokenizer.encode(text, max_length=128, truncation=True)))
            self.categories.append(category)
            self.labels.append(label)

        assert len(self.texts) == len(self.labels)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx], self.categories[idx]


class packDataset_util_bert_with_category:
    def __init__(self):
        pass

    def fn(self, data):
        texts = []
        labels = []
        categories = []
        for text, category, label in data:
            texts.append(text)
            labels.append(label)
            categories.append(category)
        labels = torch.tensor(labels)
        categories = torch.tensor(categories)
        padded_texts = pad_sequence(texts, batch_first=True, padding_value=0)
        attention_masks = torch.zeros_like(padded_texts).masked_fill(padded_texts != 0, 1)
        return padded_texts, attention_masks, labels, categories

    def get_loader(self, data, shuffle=True, batch_size=32):
        dataset = processed_dataset_bert_with_category(data)
        loader = DataLoader(dataset=dataset, shuffle=shuffle, batch_size=batch_size, collate_fn=self.fn)
        return loader


class processed_dataset_bert_with_all(Dataset):
    def __init__(self, data):
        self.texts = []
        self.categories = []
        self.sentiments = []
        self.tags = []
        self.hate = []
        self.sarcasm = []
        self.difficulities = []
        self.labels = []

        for text, category, sentiment, tag, difficulity, label in data:
            self.texts.append(torch.tensor(tokenizer.encode(text, max_length=128, truncation=True)))
            self.categories.append(category)
            self.sentiments.append(sentiment)
            self.tags.append(tag)
            # self.hate.append(hate)
            # self.sarcasm.append(sar)
            self.difficulities.append(difficulity)
            self.labels.append(label)

        assert len(self.texts) == len(self.labels)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.categories[idx], self.sentiments[idx], self.tags[idx], self.difficulities[idx], self.labels[idx]


class packDataset_util_bert_with_all:
    def __init__(self):
        pass

    def fn(self, data):
        texts = []
        labels = []
        categories = []
        sentiments = []
        tags = []
        hates = []
        sarcasms = []
        difficulities = []

        for text, category, sentiment, tag, difficulity, label in data:
            texts.append(text)
            labels.append(label)
            categories.append(category)
            sentiments.append(sentiment)
            tags.append(tag)
            # hates.append(hate)
            # sarcasms.append(sarcasm)
            difficulities.append(difficulity)

        labels = torch.tensor(labels)
        categories = torch.tensor(categories)
        sentiments = torch.tensor(sentiments)
        tags = torch.tensor(tags)
        # hates = torch.tensor(hates)
        # sarcasms = torch.tensor(sarcasms)
        difficulities = torch.tensor(difficulities)
        padded_texts = pad_sequence(texts, batch_first=True, padding_value=0)
        attention_masks = torch.zeros_like(padded_texts).masked_fill(padded_texts != 0, 1)
        return padded_texts, attention_masks, categories, sentiments, tags, difficulities, labels

    def get_loader(self, data, shuffle=True, batch_size=32):
        dataset = processed_dataset_bert_with_all(data)
        loader = DataLoader(dataset=dataset, shuffle=shuffle, batch_size=batch_size, collate_fn=self.fn)
        return loader
