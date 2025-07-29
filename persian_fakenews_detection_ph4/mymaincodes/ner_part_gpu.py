from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers.modeling_outputs import TokenClassifierOutput
from transformers import AutoModel
from TorchCRF import CRF
import torch.nn as nn
import numpy as np
import pickle
import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_pkl(path):
    with open(path + '.pkl', 'rb') as f:
        return pickle.load(f)


class PARSBERT(torch.nn.Module):
    def __init__(self, config, name_model):
        super(PARSBERT, self).__init__()
        self.config = config
        self.num_labels = config.num_labels
        self.parsbert = AutoModel.from_pretrained(name_model).to(device)  # Move to GPU
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels).to(device)  # Move to GPU
        self.crf = CRF(config.num_labels, use_gpu=torch.cuda.is_available())

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        token_type_ids = token_type_ids.to(device)
        labels = labels.to(device)

        outputs = self.parsbert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, **kwargs
        )
        sequence_output = self.dropout(outputs[0])
        logits = self.classifier(sequence_output)
        loss = None

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(
            loss=loss, logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )


class PARSBERTCRF(torch.nn.Module):
    def __init__(self, config, name_model):
        super(PARSBERTCRF, self).__init__()
        self.config = config
        self.num_labels = config.num_labels
        self.parsbert = AutoModel.from_pretrained(name_model)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(config.num_labels, use_gpu=True)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        outputs = self.parsbert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, **kwargs
        )
        sequence_output = self.dropout(outputs[0])
        logits = self.classifier(sequence_output)
        Z = self.crf.viterbi_decode(logits, attention_mask)
        outputs = (Z,) + outputs[2:]

        if labels is not None:
            L = self.crf.forward(logits, labels, attention_mask)
            loss = torch.sum(L)
            outputs = (-1 * loss,) + outputs

        return outputs


def tokenize_and_pad_text(sentences, tokenizer, max_len):
    tokenized_sentences = []
    tokenized_sentences_ids = []
    lens = []
    original_sentences = []
    starts = []

    for i in range(len(sentences)):
        tokenized_sentence = []
        orig_sen = []
        sentence = sentences[i].split(' ')
        start = []

        for word in sentence:
            tokenized_word = tokenizer.tokenize(word)
            if len(tokenized_word) == 1:
                start.append(1)
                if tokenized_word[0] != '[UNK]':
                    orig_sen.append(tokenized_word[0])
                else:
                    orig_sen.append(word)
            elif len(tokenized_word) > 0:
                start.append(1)
                for k in range(len(tokenized_word) - 1):
                    start.append(0)
                if '[UNK]' in tokenized_word:
                    orig_sen.extend([word] * len(tokenized_word))
                else:
                    orig_sen.extend(tokenized_word)
            tokenized_sentence.extend(tokenized_word)

        original_sentences.append(orig_sen)
        starts.append(start)

        if len(tokenized_sentence) > max_len - 2:
            # print('Warning : Size', len(tokenized_sentence), f' is bigger than max_len-2 ({max_len - 2}) , truncating index', i)
            tokenized_sentence = tokenized_sentence[:max_len - 2]

        lens.append(len(tokenized_sentence))
        tokenized_sentence = ["[CLS]"] + tokenized_sentence + ["[SEP]"]
        tokenized_sentence.extend(["[PAD]"] * (max_len - len(tokenized_sentence)))
        tokenized_sentence_id = tokenizer.convert_tokens_to_ids(tokenized_sentence)
        tokenized_sentences.append(tokenized_sentence)
        tokenized_sentences_ids.append(tokenized_sentence_id)
    return (np.array(tokenized_sentences_ids, dtype='long'),
            np.array(lens, dtype='long'), tokenized_sentences, original_sentences, starts)


def join_bpe_split_tokens(tokens, label_indices, dict2, original_sentence, starts):
    new_tokens, new_labels = [], []
    final_tokens = []
    for token, label_idx, orig_tok, start in zip(tokens, label_indices, original_sentence, starts):
        if start == 0:
            if token.startswith('##'):
                new_tokens[-1] = new_tokens[-1] + token[2:]
            else:
                new_tokens[-1] = new_tokens[-1] + token
        else:
            new_labels.append(dict2[label_idx + 1])
            new_tokens.append(token)
            final_tokens.append(orig_tok)
    return new_tokens, new_labels, final_tokens


class transformertagger():
    def __init__(self, model, tokenizer, device=device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.final_layer_is_crf = True
        if self.device.type == "cuda":
            self.model.cuda()

        self.model.eval()

    def get_label(self, seqs, dict2, bs=128):
        input_ids, lens, tokenized, original_sentences, starts = tokenize_and_pad_text(seqs, self.tokenizer, 512)
        attention_masks = [[i < lens[j] + 2 for i in range(len(ii))] for j, ii in enumerate(input_ids)]
        input_ids = input_ids.astype('int64')

        val_inputs = torch.tensor(input_ids).to(self.device)
        val_masks = torch.tensor(attention_masks).to(self.device)

        valid_data = TensorDataset(val_inputs, val_masks)
        valid_sampler = SequentialSampler(valid_data)
        valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=bs)

        predictions, true_labels = [], []

        for batch in valid_dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            b_input_ids, b_input_mask = batch

            with torch.no_grad():
                outputs = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

            if type(outputs[0]) == list:
                logits = np.array(outputs[0])
            else:
                logits = outputs[0].detach().cpu().numpy()

            if not self.final_layer_is_crf:
                predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
            else:
                # CRf does not need argmax
                predictions.extend([list(p) for p in logits])

        final_toks = []
        final_labels = []

        for i in range(len(tokenized)):
            toks, lbs, final = join_bpe_split_tokens(
                tokenized[i][1:lens[i] + 1], predictions[i][1:lens[i] + 1], dict2,
                original_sentences[i], starts[i]
            )
            final_toks.append(final)
            final_labels.append(lbs)

        return final_toks, final_labels
