from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig, AutoModel
from aggall import get_class, get_preds_with_all, number_to_one_hot, BERTModelExtrFeaturesWithLinear, update_tags
from difficulty import get_text_readability
from ner_part import transformertagger, PARSBERTCRF, load_pkl
from parts.code.PackDataset import packDataset_util_bert, packDataset_util_bert_with_all
import torch


bert_type = "HooshvareLab/bert-base-parsbert-uncased"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
packDataset_util = packDataset_util_bert(bert_type)
model_categorization = AutoModelForSequenceClassification.from_pretrained(bert_type, num_labels=11).to(device)
model_categorization.load_state_dict(torch.load("parts/code/persicacheckpoint.pth", map_location=torch.device('cuda')), strict=False)
model_sentiment = AutoModelForSequenceClassification.from_pretrained(bert_type, num_labels=2).to(device)
model_sentiment.load_state_dict(torch.load("parts/code/snappfoodCheckpoint.pth", map_location=torch.device('cuda')), strict=False)
packDataset_util_all = packDataset_util_bert_with_all(bert_type)

# additional_features = len(categories) +len(sentiments) + len(entities) + len(difficulties)
additional_features = 11 + 2 + 8 + 12
model = BERTModelExtrFeaturesWithLinear(
    bert_model=AutoModel.from_pretrained(bert_type),
    num_labels=2,
    additional_features=additional_features,
    linear_output_size=256
).to(device)
model.load_state_dict(torch.load('parts/code/aggregate_model.pth'))

data_path = 'parts/code/Peyma'
idx2tag = load_pkl(data_path + '/dict2')
tag2idx = load_pkl(data_path + '/dict_rev2')
num_classes = len(idx2tag)
model_name = 'HooshvareLab/bert-base-parsbert-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(
    model_name,
    num_labels=num_classes,
    id2label=idx2tag, label2id=tag2idx
)
# parsbert_model = PARSBERT(config, model_name).to(device)
path = "parts/code/parsbert_peyma_crf.pth"
parsbert_model = torch.load(path)
parsbert_model.eval()
tagger_obj = transformertagger(parsbert_model, tokenizer, device=device)
BATCH_SIZE = 16

min_diffs = [-3.5500000000000007, -417.428, -189.94764785553048, 8.0, 5.0, 0.0]
max_diffs = [217.5384650112867, 58.2225, 254.96799999999996, 189.6605, 451.1264108352145, 47.0]
normalize_diffs = True
entities = ['B-PCT', 'B-PER', 'B-LOC', 'B-DAT', 'B-MON', 'B-TIM', 'B-ORG', 'O']


def classify_fake(posts):
    texts_with_x = []
    for tweet in posts:
        texts_with_x.append((tweet['_source']['text'], 5))

    loader_poison = packDataset_util.get_loader(texts_with_x, shuffle=False, batch_size=BATCH_SIZE)
    categories = get_class(loader_poison, model_categorization)
    sentiments = get_class(loader_poison, model_sentiment)

    page_data = []

    for text_with_x, category, sentiment in zip(texts_with_x, categories, sentiments):
        text = text_with_x[0]
        category = number_to_one_hot(category, 11)
        sentiment = number_to_one_hot(sentiment, 2)
        ner = tagger_obj.get_label([text], idx2tag)[-1][0]
        ner_counts = []

        for entity in entities:
            ner_counts.append(ner.count(entity))

        diff = get_text_readability(text)

        for j in range(-6, 0, 1):
            min_diffs[j] = min(min_diffs[j], diff[j])
            max_diffs[j] = max(max_diffs[j], diff[j])

        page_data.append([text, 5, category, sentiment, ner_counts, diff])

    if normalize_diffs:
        for pd in page_data:
            for j in range(-6, 0, 1):
                pd[-1][j] = (pd[-1][j] - min_diffs[j]) / (max_diffs[j] - min_diffs[j])

    loader = packDataset_util_all.get_loader(page_data, shuffle=False, batch_size=BATCH_SIZE)
    preds, certainties = get_preds_with_all(loader, model)
    return preds, certainties
