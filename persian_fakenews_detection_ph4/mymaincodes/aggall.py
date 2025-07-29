import time
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
categories = ['آموزشی', 'اجتماعی', 'تاریخی', 'اقتصادی', 'بهداشتی', 'علمی', 'سیاسی', 'فرهنگی', 'فقه و حقوق', 'مذهبی',
              'ورزشی']
sentiments = ['SAD', 'HAPPY']


class BERTModelExtrFeaturesWithLinear(nn.Module):
    def __init__(self, bert_model, num_labels, additional_features, linear_output_size, temperature=12.5):
        super(BERTModelExtrFeaturesWithLinear, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.linear_layer = nn.Linear(self.bert.config.hidden_size, linear_output_size)
        self.classifier = nn.Linear(linear_output_size + additional_features, num_labels)
        self.additional_features = additional_features
        self.temperature = temperature
        self.softmax = nn.Softmax()

    def forward(self, input_ids, attention_mask, additional_input1, additional_input2, additional_input3,
                additional_input4):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        linear_output = self.linear_layer(pooled_output)
        linear_output = F.relu(linear_output)
        linear_output = self.dropout(linear_output)
        combined_output = torch.cat(
            (linear_output, additional_input1.to(torch.float32), additional_input2.to(torch.float32), additional_input3.to(torch.float32), additional_input4.to(torch.float32)), dim=1)
        logits = self.classifier(combined_output)
        logit_softmax = self.softmax(logits / self.temperature)
        return logit_softmax


def number_to_one_hot(number, size):
    one_hot = np.zeros(size)
    one_hot[number] = 1
    return one_hot


def get_class(loader, model, probability=False):
    # used for both sentiment and category
    model.eval()
    preds = []
    probs = []
    with torch.no_grad():
        for i, (padded_text, attention_masks, labels) in enumerate(loader):
            padded_text = padded_text.to(device)
            attention_masks = attention_masks.to(device)
            output = model(padded_text, attention_masks)[0]  # batch_size, 2
            _, flag = torch.max(output, dim=1)
            tmp = flag.cpu().detach().numpy()

            for k in tmp:
                preds.append(k)

                if len(preds) % 100 == 0:
                    print(len(preds), end=',')

            if probability:
                ol = output.tolist()

                for k in ol:
                    probs.append(k)

        if probability:
            return preds, probs
        else:
            return preds


def get_preds_with_all(loader, model):
    # model.eval()
    preds = []
    certainty = []
    print('start prediction:')

    with torch.no_grad():
        for i, (padded_text, attention_masks, categories, sentiments, tags, difficulities, labels) in enumerate(loader):
            padded_text = padded_text.to(device)
            attention_masks = attention_masks.to(device)
            categories = categories.to(device)
            sentiments = sentiments.to(device)
            tags = tags.to(device)
            difficulities = difficulities.to(device)
            output = model(padded_text, attention_masks, categories, sentiments, tags, difficulities)  # batch_size, 2
            certs, flag = torch.max(output, dim=1)
            tmp = flag.cpu().detach().numpy()

            for k, o in zip(tmp, output):
                preds.append(int(1 - k))  # real is equal to 1
                certainty.append(float(o[0]))

            print(i, end=',')

        print()
        return preds, certainty


def update_tags(es, index_name, page, preds, certainties):
    for tweet, pred, certainty in zip(page['hits']['hits'], preds, certainties):
        tweet['_source']['is_real'] = {"pred": pred, "certainty": certainty, "execution_time": time.time()}
        es.index(index=index_name, id=tweet['_id'], body=tweet['_source'])


def put_tags_at_first(es, index_name, news, preds, certainties):
    for new, pred, certainty in zip(news, preds, certainties):
        new['is_real'] = {"pred": pred, "certainty": certainty, "execution_time": time.time()}
        es.index(index=index_name, body=new)
