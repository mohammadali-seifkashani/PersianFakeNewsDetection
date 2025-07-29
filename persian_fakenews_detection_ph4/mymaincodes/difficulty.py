import math
import hazm
import pandas as pd


class PersianSyllableCounter:
    def __init__(self):
        self.vowels = ('a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U')
        self.persian_phonetics_dictionary = {}
        self.persian_syllables_dictionary = {}
        self.persian_syllables_basedOnLength_dictionary = {}
        self.initialize_dictionaries()

    def count_syllables_in_word(self, word):
        ans = 0
        for letter in word:
            if letter in self.vowels:
                ans += 1
        return ans

    def initialize_dictionaries(self):
        f = open('mymaincodes/Peyma/persian_phonetics.txt', encoding="utf8")
        lines = f.readlines()
        for line in lines:
            word, phonetic = line.split()
            self.persian_phonetics_dictionary[word] = phonetic
            word_syllables = self.count_syllables_in_word(phonetic)
            self.persian_syllables_dictionary[word] = word_syllables
            word_len = len(word)
            if word_len in self.persian_syllables_basedOnLength_dictionary:
                self.persian_syllables_basedOnLength_dictionary[word_len].append(word_syllables)
            else:
                self.persian_syllables_basedOnLength_dictionary[word_len] = [word_syllables]

        for word_len in self.persian_syllables_basedOnLength_dictionary.keys():
            lengths = self.persian_syllables_basedOnLength_dictionary[word_len]
            self.persian_syllables_basedOnLength_dictionary[word_len] = round(sum(lengths)/ len(lengths))

    def pridict_syllable(self, word):
        word_len = len(word)
        if word_len == 1:
            return 1
        if word_len in self.persian_syllables_basedOnLength_dictionary:
            return self.persian_syllables_basedOnLength_dictionary[word_len]
        return self.pridict_syllable(word[1:])

    def count_syllables_in_text(self, text):
        ans = 0
        splitted_text = text.split()
        for word in splitted_text:
            if word in self.persian_syllables_dictionary:
                ans += self.persian_syllables_dictionary[word]
            else:
                ans += self.pridict_syllable(word)
        return ans


class utils:
    SPECIAL_CHARS = ['.', ',', '!', '?', '،']
    PERSIAN_NUMBERS = ['۰', '۱', '۲', '۳', '۴', '۵', '۶', '۷', '۸', '۹']
    psc = PersianSyllableCounter()

    @staticmethod
    def get_sentences(text):
        sentences = hazm.sent_tokenize(text)
        return sentences

    @staticmethod
    def get_words(text='',normalizer = hazm.Normalizer()):
        normalizedText = normalizer.normalize(text)
        words = hazm.word_tokenize(normalizedText)
        filtered_words = []
        for word in words:
            if word in utils.SPECIAL_CHARS or word == " ":
                pass
            else:
                new_word = word.replace(",","").replace(".","")
                new_word = new_word.replace("!","").replace("?","")
                filtered_words.append(new_word)
        return filtered_words

    @staticmethod
    def get_char_count(words):
        characters = 0
        for word in words:
            characters += len(word)
        return characters

    @staticmethod
    def count_syllables(words):
        syllableCount = 0
        for word in words:
            syllableCount += utils.psc.count_syllables_in_text(word)
        return syllableCount

    @staticmethod
    def count_complex_words(text):
        words = utils.get_words(text)
        sentences = utils.get_sentences(text)
        complex_words = 0
        found = False
        cur_word = []

        for word in words:
            cur_word.append(word)
            if utils.count_syllables(cur_word)>= 3:
                complex_words += 1
                #Checking proper nouns
                # TODO

                # if not(word[0].isupper()):
                #     complex_words += 1
                # else:
                #     for sentence in sentences:
                #         if str(sentence).startswith(word):
                #             found = True
                #             break
                #     if found:
                #         complex_words += 1
                #         found = False
            cur_word.remove(word)
        return complex_words


class readability :
    analyzedVars = {}

    def __init__(self, text):
        self.analyze_text(text)

    def analyze_text(self, text):
        words = utils.get_words(text)
        char_count = utils.get_char_count(words)
        word_count = len(words)
        sentence_count = len(utils.get_sentences(text))
        syllable_count = utils.count_syllables(words)
        complexwords_count = utils.count_complex_words(text)
        avg_words_p_sentence = word_count/sentence_count

        self.analyzedVars = {
            'words': words,
            'char_cnt': float(char_count),
            'word_cnt': float(word_count),
            'sentence_cnt': float(sentence_count),
            'syllable_cnt': float(syllable_count),
            'complex_word_cnt': float(complexwords_count),
            'avg_words_p_sentence': float(avg_words_p_sentence)
        }

    def ARI(self):
        score = 0.0
        if self.analyzedVars['word_cnt'] > 0.0:
            score = 4.71 * (self.analyzedVars['char_cnt'] / self.analyzedVars['word_cnt']) + 0.5 * (self.analyzedVars['word_cnt'] / self.analyzedVars['sentence_cnt']) - 21.43
        return score

    def FleschReadingEase(self):
        score = 0.0
        if self.analyzedVars['word_cnt'] > 0.0:
            score = 206.835 - (1.015 * (self.analyzedVars['avg_words_p_sentence'])) - (84.6 * (self.analyzedVars['syllable_cnt']/ self.analyzedVars['word_cnt']))
        return round(score, 4)

    def FleschKincaidGradeLevel(self):
        score = 0.0
        if self.analyzedVars['word_cnt'] > 0.0:
            score = 0.39 * (self.analyzedVars['avg_words_p_sentence']) + 11.8 * (self.analyzedVars['syllable_cnt']/ self.analyzedVars['word_cnt']) - 15.59
        return round(score, 4)

    def FleschDayaniReadability(self):
        score = 0.0
        if self.analyzedVars['word_cnt'] > 0.0:
            score = ((262.835) - (0.846 * (self.analyzedVars['char_cnt'] / self.analyzedVars['word_cnt']))) - (1.015 * (self.analyzedVars['word_cnt'] / self.analyzedVars['sentence_cnt']))
        return score

    def GunningFogIndex(self):
        score = 0.0
        if self.analyzedVars['word_cnt'] > 0.0:
            score = 0.4 * ((self.analyzedVars['avg_words_p_sentence']) + (100 * (self.analyzedVars['complex_word_cnt']/self.analyzedVars['word_cnt'])))
        return round(score, 4)

    def SMOGIndex(self):
        score = 0.0
        if self.analyzedVars['word_cnt'] > 0.0:
            score = (math.sqrt(self.analyzedVars['complex_word_cnt']*(30/self.analyzedVars['sentence_cnt'])) + 3)
        return score

    def ColemanLiauIndex(self):
        score = 0.0
        if self.analyzedVars['word_cnt'] > 0.0:
            score = (5.89*(self.analyzedVars['char_cnt']/self.analyzedVars['word_cnt']))-(30*(self.analyzedVars['sentence_cnt']/self.analyzedVars['word_cnt']))-15.8
        return round(score, 4)

    def LIX(self):
        longwords = 0.0
        score = 0.0
        if self.analyzedVars['word_cnt'] > 0.0:
            for word in self.analyzedVars['words']:
                if len(word) >= 7:
                    longwords += 1.0
            score = self.analyzedVars['word_cnt'] / self.analyzedVars['sentence_cnt'] + float(100 * longwords) / self.analyzedVars['word_cnt']
        return score

    def RIX(self):
        longwords = 0.0
        score = 0.0
        if self.analyzedVars['word_cnt'] > 0.0:
            for word in self.analyzedVars['words']:
                if len(word) >= 7:
                    longwords += 1.0
            score = longwords / self.analyzedVars['sentence_cnt']
        return score


def get_text_readability(text):
    rd = readability(text)
    row_data = {
        'char_count': rd.analyzedVars['char_cnt'],
        'word_count': rd.analyzedVars['word_cnt'],
        'sentence_count': rd.analyzedVars['sentence_cnt'],
        'syllable_count': rd.analyzedVars['syllable_cnt'],
        'complex_word_count': rd.analyzedVars['complex_word_cnt'],
        'average_words_per_sentence': rd.analyzedVars['avg_words_p_sentence'],
        'ARI': rd.ARI(),
        'FleschReadingEase': 1 - rd.FleschReadingEase(),
        'FleschDayaniReadability': 1 - rd.FleschDayaniReadability(),
        # 'FleschKincaidGradeLevel': rd.FleschKincaidGradeLevel(),
        'GunningFogIndex': rd.GunningFogIndex(),
        # 'SMOGIndex': rd.SMOGIndex(),
        # 'ColemanLiauIndex': rd.ColemanLiauIndex(),
        'LIX': rd.LIX(),
        'RIX': rd.RIX()
    }

    return list(row_data.values())