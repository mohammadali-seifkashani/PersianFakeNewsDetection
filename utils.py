from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timedelta
from time import time, sleep
import numpy as np
import jdatetime
import psutil
import pytz
import json
import time
import re
import os


PHONE_REGEX = re.compile(
    r"((?:^|(?<=[^\w)]))(((\+?[01])|(\+\d{2}))[ .-]?)?(\(?\d{3,4}\)?/?[ .-]?)?(\d{3}[ .-]?\d{4})(\s?(?:ext\.?|[#x-])\s?\d{2,6})?(?:$|(?=\W)))|\+?\d{4,5}[ .-/]\d{6,9}"
)
URL_REGEX = re.compile(r'https?://\S+|www\.\S+')
MENTION_REGEX = re.compile(r'@\w+')
RE_EMOJI = re.compile(r'[\U00010000-\U0010ffff]', flags=re.UNICODE)
ENGLISH_PERSIAN = r'[^a-zA-Z\u0600-\u06FF0-9\u06F0-\u06F9\s./,!?؛؟()\-#:_]'


def remove_emoji_phone_link_mention(text):
    text = URL_REGEX.sub('', text)
    text = PHONE_REGEX.sub('', text)
    text = MENTION_REGEX.sub('', text)
    # text = RE_EMOJI.sub('', text)
    text = re.sub(ENGLISH_PERSIAN, ' ', text)
    return text.strip()


def convert_persian_to_english_numbers(text):
    text = str(text)
    return ''.join(chr(ord(char) - 1728) if '۰' <= char <= '۹' else char for char in text)


def convert_english_to_persian_numbers(text):
    text = str(text)
    return ''.join(chr(ord(char) + 1728) if '0' <= char <= '9' else char for char in text)


def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss


def ExecuteQuery(connection, query):
    with connection.connect() as conn:
        conn.execute(query)
        conn.commit()
        conn.close()


def time_decorator(func):
    def inner(*args, **kwargs):
        start_time = time()
        out = func(*args, **kwargs)
        spent_time = time() - start_time
        print(f"spent time in {func.__name__}():", spent_time)
        return out

    return inner


def lf_utc_time_to_persian(utc_time_str):
    utc_time = datetime.strptime(utc_time_str, '%Y-%m-%dT%H:%M:%SZ')
    utc_time = pytz.utc.localize(utc_time)
    iran_time = utc_time.astimezone(pytz.timezone('Asia/Tehran'))
    persian_datetime = jdatetime.datetime.fromgregorian(datetime=iran_time)
    return persian_datetime


def save_json(data, file_name):
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=3)


def load_json(file_name):
    with open(file_name, 'r', encoding='utf-8') as json_file:
        return json.load(json_file)


def sleep_counting(s):
    for i in range(1, s + 1):
        sleep(1)
        print(i, end=',')
    print()


def two_texts_similarity(t1, t2):
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer
    import torch

    def get_text_chunks_embeddings(embed_model, model_tokenizer, text):
        max_tokens_per_chunk = 510  # Adjust as needed
        paragraph_tokens = 0
        chunk_embeddings = []
        chunks = [[]]

        for line in text.split('\n'):
            line_tokens = model_tokenizer.tokenize(line, max_length=max_tokens_per_chunk, truncation=False)
            if paragraph_tokens + len(line_tokens) > max_tokens_per_chunk:
                chunks.append([])
                paragraph_tokens = 0
            paragraph_tokens += len(line_tokens)
            chunks[-1] += line_tokens

        for chunk_tokens in chunks:
            chunk_text = model_tokenizer.convert_tokens_to_string(chunk_tokens)
            chunk_embedding = embed_model.encode([chunk_text])
            chunk_embeddings.append(chunk_embedding)
        return chunk_embeddings

    def get_text_mean_embedding(embed_model, model_tokenizer, text):
        try:
            chunk_embeddings = get_text_chunks_embeddings(embed_model, model_tokenizer, text)
        except RuntimeError as e:
            print(f'\n{e}')
            print('len(text):', len(text))
            print('sleeping for 5s ...')
            sleep(5)
            return get_text_mean_embedding(embed_model, model_tokenizer, text)
        return np.mean(chunk_embeddings, axis=0)[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = 'sentence-transformers/LaBSE'
    embed_model = SentenceTransformer(model_name, device=device)
    model_tokenizer = AutoTokenizer.from_pretrained(model_name)
    em1 = get_text_mean_embedding(embed_model, model_tokenizer, remove_emoji_phone_link_mention(t1))
    em2 = get_text_mean_embedding(embed_model, model_tokenizer, remove_emoji_phone_link_mention(t2))
    return cosine_similarity([em1], [em2])[0]


class ElasticUtils:
    def __init__(self, es):
        self.es = es

    def get_all_indices(self):
        indices = self.es.cat.indices(format="json")
        index_names = [index['index'] for index in indices if not index['index'].startswith('.')]
        return index_names

    def search_elastic_query(self, index_name, query, scroll_size=10000):
        try:
            page = self.es.search(
                index=index_name,
                scroll='30m',  # Length of time to keep the scroll window open
                size=scroll_size,  # Number of results to return per scroll
                body={
                    "query": query
                }
            )
        except Exception as e:
            print(f'Exception in search_elastic_query() function: {e}')
            sleep_counting(30)
            return self.search_elastic_query(index_name, query)
        # total_hits = page['hits']['total']['value']
        sid = page['_scroll_id']
        posts = page['hits']['hits']
        scroll_size = len(posts)

        while scroll_size > 0:
            try:
                page = self.es.scroll(scroll_id=sid, scroll='30m')
                sid = page['_scroll_id']
            except Exception as e:
                print(f'Elastic Exception in search_elastic_query() in while part: {e}')
                sleep_counting(30)
                continue

            posts += page['hits']['hits']
            scroll_size = len(page['hits']['hits'])

        return posts

    def search_elastic_query_yield(self, index_name, query, scroll_size=10000):
        try:
            page = self.es.search(
                index=index_name,
                scroll='30m',  # Length of time to keep the scroll window open
                size=scroll_size,  # Number of results to return per scroll
                body={
                    "query": query
                }
            )
        except Exception as e:
            print(f'Exception in search_elastic_query_yield() function: {e}')
            sleep_counting(30)
            return self.search_elastic_query_yield(index_name, query)
        # total_hits = page['hits']['total']['value']
        sid = page['_scroll_id']
        scroll_size = len(page['hits']['hits'])
        yield page['hits']['hits']

        while scroll_size > 0:
            try:
                page = self.es.scroll(scroll_id=sid, scroll='30m')
                sid = page['_scroll_id']
            except Exception as e:
                print(f'Elastic Exception in search_elastic_query_yield() function: {e}')
                sleep_counting(30)
                continue

            scroll_size = len(page['hits']['hits'])
            yield page['hits']['hits']

    def delete_elastic_query(self, index_name, query):
        self.es.delete_by_query(index=index_name, body={'query': query})

    def remove_field_from_all_posts(self, index_name, field):
        query = {
            "bool": {
                "filter": [{"exists": {"field": field}}],
            }
        }
        posts = self.search_elastic_query(index_name, query)
        for i, p in enumerate(posts):
            p['_source'].pop(field, None)
            self.es.index(index=index_name, id=p['_id'], body=p['_source'])
            if i % 100 == 0 and i != 0:
                print(i, end=',')

    def remove_embeddings(self, index_name='daily_trends_posts', days=10):
        xdaysbefore = datetime.now().date() - timedelta(days=days)
        query = {
            "bool": {
                "must": [
                    {"exists": {"field": "embedding"}},
                    {"range": {"published_at": {"lte": xdaysbefore}}}
                ]
            }
        }
        posts = self.search_elastic_query(index_name, query)

        for i, p in enumerate(posts):
            p['_source'].pop('embedding', None)
            self.es.index(index=index_name, id=p['_id'], body=p['_source'])
            if i % 100 == 0 and i != 0:
                print(i, end=',')


if __name__ == '__main__':
    print()