from transformers import AutoModelForSequenceClassification, AutoModel, AutoTokenizer, AutoConfig
from mymaincodes.aggall import get_class
from mymaincodes.ner_part_gpu import load_pkl, PARSBERTCRF, transformertagger
from utils import load_json, save_json, remove_emoji_phone_link_mention
from mymaincodes.PackDataset import packDataset_util_bert
from mymaincodes.difficulty import get_text_readability
import pandas as pd
import numpy as np
import os
import re
from sklearn.preprocessing import MinMaxScaler
from utils import load_json, save_json, es, EU, sleep_counting, remove_emoji_phone_link_mention
import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
folder_path = '/home/seif/projects/momtazi/phase4'
idx2tag = load_pkl(folder_path + '/mymaincodes/Peyma/dict2')
tag2idx = load_pkl(folder_path + '/mymaincodes/Peyma/dict_rev2')
BATCH_SIZE = 64
normalize_diffs = True
entities = ['B-PCT', 'B-PER', 'B-LOC', 'B-DAT', 'B-MON', 'B-TIM', 'B-ORG', 'O']
bert_type = "HooshvareLab/bert-base-parsbert-uncased"
packDataset_util = packDataset_util_bert()
model_categorization = AutoModelForSequenceClassification.from_pretrained(bert_type, num_labels=11).to(device)
model_categorization.load_state_dict(torch.load(f"{folder_path}/models/persicacheckpoint.pth", map_location=device), strict=False)
model_sentiment = AutoModelForSequenceClassification.from_pretrained(bert_type, num_labels=2).to(device)
model_sentiment.load_state_dict(torch.load(f"{folder_path}/models/snappfoodCheckpoint.pth", map_location=device), strict=False)


def get_tagger_obj_v1():
    num_classes = len(idx2tag)
    model_name = 'HooshvareLab/bert-base-parsbert-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=num_classes,
        id2label=idx2tag, label2id=tag2idx
    )
    path = f"{folder_path}/models/parsbert_peyma_crf.pth"
    parsbert_model = PARSBERTCRF(config, model_name).to(device)
    parsbert_model.load_state_dict(torch.load(path, map_location=device).state_dict())
    parsbert_model.eval()
    tagger_obj = transformertagger(parsbert_model, tokenizer, device=device)
    return tagger_obj


def get_features(userembeddings, texts, labels, tagger_obj):
    texts_with_x = []

    for text in texts:
        texts_with_x.append((text, -1))

    loader_poison = packDataset_util.get_loader(texts_with_x, shuffle=False, batch_size=BATCH_SIZE)
    print('getting category and sentiment of texts:')
    catpreds, categories = get_class(loader_poison, model_categorization, probability=True)
    sentpreds, sentiments = get_class(loader_poison, model_sentiment, probability=True)
    print()
    page_data = []
    print('tagging NER and dfficulty:')

    for i, (uem, text_with_x, category, sentiment, label) in enumerate(zip(userembeddings, texts_with_x, categories, sentiments, labels)):
        text = text_with_x[0]
        # category = number_to_one_hot(category, 11).tolist()
        # sentiment = number_to_one_hot(sentiment, 2).tolist()
        ner = tagger_obj.get_label([text], idx2tag)[-1][0]
        ner_counts = []

        for entity in entities:
            ner_counts.append(ner.count(entity))

        # diff = get_text_readability(text)
        diff = get_text_readability(text)[-6:]

        page_data.append([uem, text, category, sentiment, ner_counts, diff, label])

        if i % 100 == 0 and i != 0:
            print(i, end=',')

    print()
    return page_data


def clean_text(text):
    if pd.isna(text):
        return text
    first_25 = text[:25]
    remaining_text = text[25:]
    cleaned_first_25 = re.sub(r".*گزارش.*['،؛]:?\s*", "", first_25)
    return remove_emoji_phone_link_mention(cleaned_first_25 + remaining_text)


def get_users():
    fu = pd.read_json('userdataset/faradade_userinfo.json').transpose()
    accouninfo = load_json('userdataset/account_info.json')
    reliabilities = load_json('userdataset/user_certainty_history.json')
    fu = fu[['Gender', 'Age', 'PoliticalParty', 'Religion', 'Job', 'Faith']]
    fu['followers'] = 0
    fu['followings'] = 0
    fu['reliability'] = 0
    # fu['post_count'] = 0

    for col in fu.select_dtypes(include=['object', 'category']).columns:
        fu[col] = pd.factorize(fu[col], sort=True)[0] + 1  # Starting from 1
        fu[col] = fu[col].replace(pd.NA, 0)  # Replace NaN back if needed

    for i, row in fu.iterrows():
        accountrow = accouninfo.get(row.name.lower(), {})
        mean_reliability = float(np.mean(reliabilities.get(row.name.lower(), [])))
        fu.at[i, 'followers'] = accountrow.get('followers')
        fu.at[i, 'followings'] = accountrow.get('followings')
        fu.at[i, 'reliability'] = mean_reliability

    scaler = MinMaxScaler()
    numeric_cols = fu.select_dtypes(include=['number']).columns
    fu[numeric_cols] = scaler.fit_transform(fu[numeric_cols])
    fu = fu.fillna(0)
    fu = fu[~(fu == 0).all(axis=1)]
    return fu


tagger_obj = get_tagger_obj_v1()
data = []
d = load_json('elastic/agguser.json')
uem = load_json('elastic/user10periodcertainty.json')
# users = get_users()
texts = []
labels = []
userembeddings = []

for i, record in enumerate(d):
    screen_name = record[1].lower()
    userembedding = uem[screen_name]
    text = clean_text(record[3])
    label = record[-1]

    texts.append(text)
    labels.append(label)
    userembeddings.append(userembedding)

    if i % 10 == 0:
        print(i, end=',')

data += get_features(userembeddings, texts, labels, tagger_obj)
save_json(data, 'aggallcert.json')
print(len(data))
