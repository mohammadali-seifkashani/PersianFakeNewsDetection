from sklearn.metrics import confusion_matrix, classification_report
from mymaincodes.PackDataset_ph2 import packDataset_util_bert_with_all
from sklearn.model_selection import train_test_split
from torch.nn.utils import clip_grad_norm_
from transformers import AutoModel
from sklearn.utils import shuffle
import torch.nn.functional as F
from utils import load_json
import torch.nn as nn
import transformers
import numpy as np
import random
import torch
import copy
import os


os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


random_seed = 101
set_seed(random_seed)
BATCH_SIZE = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bert_type = "HooshvareLab/bert-base-parsbert-uncased"
# bert_type = '/home/gheysari/general_sentiment/matina_llm/'
packDataset_util = packDataset_util_bert_with_all()
categories = ['آموزشی', 'اجتماعی', 'تاریخی', 'اقتصادی', 'بهداشتی', 'علمی', 'سیاسی', 'فرهنگی', 'فقه و حقوق', 'مذهبی', 'ورزشی']
sentiments = ['SAD', 'HAPPY']
# entities = ['I-ORG', 'B-MON', 'I-PCT', 'B-TIM', 'B-LOC', 'I-TIM', 'I-LOC', 'B-PER', 'B-ORG', 'B-PCT', 'O', 'I-DAT', 'I-PER', 'B-DAT', 'I-MON']
entities = ['B-MON', 'B-TIM', 'B-LOC', 'B-PER', 'B-ORG', 'B-PCT', 'O', 'B-DAT']
difficulty_B_M = ['char_count', 'word_count', 'sentence_count', 'syllable_count', 'complex_word_count', 'average_words_per_sentence', 'ARI','FleschReadingEase', 'FleschDayaniReadability', 'GunningFogIndex', 'LIX', 'RIX']
difficulty_H_M = ['ARI', 'FleschReadingEase', 'FleschDayaniReadability', 'GunningFogIndex', 'LIX', 'RIX']
min_diffs = [np.inf] * 6
max_diffs = [-np.inf] * 6
normalize_diffs_bool = True


class BERTModelExtrFeaturesWithLinear(nn.Module):
    def __init__(self, bert_model, num_labels, additional_features, linear_output_size, temperature=8):
        super(BERTModelExtrFeaturesWithLinear, self).__init__()
        self.bert = bert_model
        self.temperature = temperature
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.linear_layer = nn.Linear(self.bert.config.hidden_size, linear_output_size)
        self.classifier = nn.Linear(linear_output_size + additional_features, num_labels)
        self.additional_features = additional_features
        self.softmax = nn.Softmax()

    def forward(self, input_ids, attention_mask, additional_input1, additional_input2, additional_input3, additional_input4):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)

        linear_output = self.linear_layer(pooled_output)
        linear_output = F.relu(linear_output)
        linear_output = self.dropout(linear_output)

        combined_output = torch.cat(
            (linear_output, additional_input1, additional_input2, additional_input3, additional_input4), dim=1)
        logits = self.classifier(combined_output)
        logit_softmax = self.softmax(logits / self.temperature)
        return logit_softmax
    

def normalize_diffs(data):
    for d in data:
        diff = d[-2]
        for j in range(-6, 0, 1):
            min_diffs[j] = min(min_diffs[j], diff[j])
            max_diffs[j] = max(max_diffs[j], diff[j])
    
    for d in data:
        for j in range(-6, 0, 1):
            d[-2][j] = (d[-2][j] - min_diffs[j]) / (max_diffs[j] - min_diffs[j])



def select_data(data, class1rate=2):
    result = []
    l0 = l1 = 0

    for d in data:
        if d[-1] == 0:
            l0 += 1
        else:
            l1 += 1

    class1count = int(class1rate * l0)
    count = 0
    print('label 0:', l0)
    print('label 1:', l1)

    if class1count > l1:
        class1count = l1

    for d in data:
        if d[-1] == 0:
            result.append(d)
        elif count < class1count:
            result.append(d)
            count += 1
    
    return result


def get_train_dev_test_loaders(class1rate=2.5):
    data = load_json('aggallcert.json')
    for d in data:
        d[-1] = int(d[-1])
    data = shuffle(data, random_state=42)
    if normalize_diffs_bool:
        normalize_diffs(data)
    # data = select_data(data, class1rate)
    train_data, remaining_data = train_test_split(data, test_size=0.3, random_state=42)
    val_data, test_data = train_test_split(remaining_data, test_size=0.5, random_state=42)
    print(test_data[0])

    train_list = []
    test_list = []
    dev_list = []
    for i in range(len(train_data)):
        train_list.append(train_data[i][1:])

    for i in range(len(test_data)):
        test_list.append(test_data[i][1:])

    for i in range(len(val_data)):
        dev_list.append(val_data[i][1:])

    packDataset_util = packDataset_util_bert_with_all()
    train_loader_poison = packDataset_util.get_loader(train_list, shuffle=True, batch_size=BATCH_SIZE)
    dev_loader_poison = packDataset_util.get_loader(dev_list, shuffle=False, batch_size=BATCH_SIZE)
    test_loader_poison = packDataset_util.get_loader(test_list, shuffle=False, batch_size=BATCH_SIZE)
    return train_loader_poison, dev_loader_poison, test_loader_poison


def train(model, criterion, optimizer, scheduler, train_loader_poison, early_stopping_rounds=3):
    last_train_avg_loss = 1000000
    model_temp = model
    best_valid_acc = 0
    rounds_without_improvement = 0
    try:
        for epoch in range(warm_up_epochs + EPOCHS):
            print('epoch:', epoch)
            model.train()
            total_loss = 0
            for padded_text, attention_masks, categories, sentiments, tags, difficulity, labels in train_loader_poison:
                padded_text = padded_text.to(device)
                attention_masks = attention_masks.to(device)
                categories = categories.to(device)
                labels = labels.to(device).long()
                sentiments = sentiments.to(device)
                tags = tags.to(device)
                difficulity = difficulity.to(device)
                output = model(padded_text, attention_masks, categories, sentiments, tags, difficulity)
                try:
                    loss = criterion(output, labels)
                    optimizer.zero_grad()
                    loss.backward()
                except Exception as e:
                    print()
                    print(e)
                clip_grad_norm_(model.parameters(), max_norm=1)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader_poison)
            print('finish training, avg loss: {}/{}, begin to evaluate'.format(avg_loss, last_train_avg_loss))
            valid_acc = evaluation(dev_loader_poison)
            test_acc = evaluation(test_loader_poison)
            last_train_avg_loss = avg_loss

            if valid_acc >= best_valid_acc:
                best_valid_acc = valid_acc
                rounds_without_improvement = 0
                model_temp = copy.deepcopy(model)
            else:
                rounds_without_improvement += 1
                if rounds_without_improvement >= early_stopping_rounds:
                    print(f'No improvement for {early_stopping_rounds} epochs. Early stopping...')
                    model = model_temp
                    break
            print('*' * 89)

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    test_acc = evaluation(test_loader_poison)
    print('*' * 89)
    print(f'finish all, test acc: {test_acc}')
    return model_temp


def evaluation(loader):
    model.eval()
    total_number = 0
    total_correct = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for padded_text, attention_masks, categories, sentiments, tags, difficulity, labels in loader:
            padded_text = padded_text.to(device)
            attention_masks = attention_masks.to(device)
            categories = categories.to(device)
            sentiments = sentiments.to(device)
            tags = tags.to(device)
            difficulity = difficulity.to(device)
            labels = labels.to(device)
            output = model(padded_text, attention_masks, categories, sentiments, tags, difficulity)  # batch_size, 2
            _, flag = torch.max(output, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(flag.cpu().numpy())
            total_number += labels.size(0)
            correct = (flag == labels).sum().item()
            total_correct += correct

    print(classification_report(all_labels, all_preds, target_names=["Class 0", "Class 1"]))
    cr = classification_report(all_labels, all_preds, target_names=["Class 0", "Class 1"], output_dict=True)
    return cr['Class 0']['f1-score']


class1rate = 2.5
train_loader_poison, dev_loader_poison, test_loader_poison = get_train_dev_test_loaders(class1rate)
weight_decay = 0.0
lr = 2e-5
EPOCHS = 15
warm_up_epochs = 3
additional_features = len(categories) + len(sentiments) + len(entities) + len(difficulty_H_M)
model = BERTModelExtrFeaturesWithLinear(
    bert_model=AutoModel.from_pretrained(bert_type),
    num_labels=2,
    additional_features=additional_features,
    linear_output_size=256
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = transformers.get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warm_up_epochs * len(train_loader_poison),
    num_training_steps=(warm_up_epochs + EPOCHS) * len(train_loader_poison)
)
model = train(model, criterion, optimizer, scheduler, train_loader_poison, 10)
print(model)
evaluation(test_loader_poison)
evaluation(dev_loader_poison)
torch.save(model.state_dict(), "text14040206.pth")
