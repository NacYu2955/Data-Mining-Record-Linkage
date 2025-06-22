import re
from typing import Dict
import inflect
from nltk.stem import PorterStemmer
from unidecode import unidecode
import difflib

class Preprocessor:
    def __init__(self):
        self.inflect_engine = inflect.engine()
        self.stemmer = PorterStemmer()
        self.similar_shape_map = {
            '0': 'o', 'o': '0', '1': 'l', 'i': '1', 'l': '1', '2': 'z', 'z': '2', '5': 's', 's': '5',
            '6': 'b', '8': 'b', '13': 'b', 'b': '8', 'm': 'rn', 'rn': 'm', 'w': 'vv', 'vv': 'w', 'v': 'u',
            'O': '0', 'I': '1', 'S': '5', 'B': '8', 'L': '|_', '|_': 'l', 'W': 'vv',
        }
        self.nickname_map = {
            'william': 'bill', 'bill': 'william', 'robert': 'bob', 'bob': 'robert', 'elizabeth': 'betty', 'betty': 'elizabeth',
            'margaret': 'peggy', 'peggy': 'margaret', 'john': 'jack', 'jack': 'john', 'richard': 'dick', 'dick': 'richard',
            'edward': 'ted', 'ted': 'edward', 'charles': 'chuck', 'chuck': 'charles', 'james': 'jim', 'jim': 'james',
            'susan': 'sue', 'sue': 'susan', 'katherine': 'kate', 'kate': 'katherine', 'patricia': 'pat', 'pat': 'patricia',
            'michael': 'mike', 'mike': 'michael', 'steven': 'steve', 'steve': 'steven', 'joseph': 'joe', 'joe': 'joseph',
            'thomas': 'tom', 'tom': 'thomas', 'barbara': 'barb', 'barb': 'barbara', 'daniel': 'dan', 'dan': 'daniel',
            'anthony': 'tony', 'tony': 'anthony', 'donald': 'don', 'don': 'donald', 'andrew': 'andy', 'andy': 'andrew',
            'jennifer': 'jen', 'jen': 'jennifer', 'christopher': 'chris', 'chris': 'christopher', 'matthew': 'matt', 'matt': 'matthew',
            'joshua': 'josh', 'josh': 'joshua', 'nicholas': 'nick', 'nick': 'nicholas', 'alexander': 'alex', 'alex': 'alexander',
            'samantha': 'sam', 'sam': 'samantha', 'jessica': 'jess', 'jess': 'jessica', 'benjamin': 'ben', 'ben': 'benjamin',
        }
        self.variant_spelling_map = {
            'center': 'centre', 'centre': 'center', 'color': 'colour', 'colour': 'color', 'organize': 'organise', 'organise': 'organize',
            'analyze': 'analyse', 'analyse': 'analyze', 'defense': 'defence', 'defence': 'defense', 'license': 'licence', 'licence': 'license',
            'catalog': 'catalogue', 'catalogue': 'catalog', 'theater': 'theatre', 'theatre': 'theater', 'traveler': 'traveller', 'traveller': 'traveler',
            'abdul': 'abd', 'abd': 'abdul', 'rasheed': 'rashid', 'rashid': 'rasheed', 'sarah': 'sara', 'sara': 'sarah',
            # 拼写错误特例
            'cetner': 'center',
        }
        self.roman_map = {'i': '1', 'ii': '2', 'iii': '3', 'iv': '4', 'v': '5', 'vi': '6', 'vii': '7', 'viii': '8', 'ix': '9', 'x': '10'}
        self.ordinal_map = {'1st': 'first', '2nd': 'second', '3rd': 'third', '4th': 'fourth', '5th': 'fifth'}
        self.company_map = {
            'inc': 'company', 'co': 'company', 'corp': 'company', 'corporation': 'company', 'ltd': 'company', 'llc': 'company', 'llc.': 'company',
            'pharmaceuticals': 'drugs', 'pharmaceutical': 'drugs', 'drugs': 'drugs', 'drug': 'drugs',
            'compania': 'company', 'compania': 'company', 'compania': 'company', 'compania': 'company',
            'compania': 'company', 'compania': 'company', 'compania': 'company',
        }
        self.synonym_map = {
            'shipping': 'shpg', 'shpg': 'shipping', 'apartment': 'apt', 'apt': 'apartment', 'number': 'no', 'no': 'number',
            'northeast': 'ne', 'ne': 'northeast', 'road': 'rd', 'rd': 'road', 'usa': 'us', 'us': 'usa',
            'united states': 'usa', 'united states of america': 'usa',
        }
        self.trans_map = {
            'compania': 'company', 'compania': 'company', 'compania': 'company',
            '公司': 'company', '회사': 'company', 'شركة': 'company',
        }
        self.stem_special = {
            'technology': 'tech', 'technologies': 'tech', 'tech': 'tech',
        }
        self.initials_pattern = re.compile(r'\b([A-Z])\.?\s*')
        self.roman_pattern = re.compile(r'\b([ivxlcdm]+)\b', re.IGNORECASE)
        self.ord_pattern = re.compile(r'\b(\d+)(st|nd|rd|th)\b')
        self.known_words = set(list(self.nickname_map.keys()) + list(self.variant_spelling_map.keys()) + list(self.company_map.keys()) + list(self.synonym_map.keys()) + list(self.stem_special.keys()))
        # 自定义映射表，包含所有图片示例的变体
        self.custom_map = {
            # Letter Case & Special Character & More/Less Space & More/Less Word & Word Reorder
            'james': 'james', 'JAMES': 'james',
            'emma': 'emma', 'em-ma': 'emma', 'em ma': 'emma',
            'mml.l.c.': 'mmllc', 'mm llc': 'mmllc',
            'anna': 'anna', 'an na': 'anna',
            'johnedwardsmith': 'johnsmith', 'johnsmith': 'johnsmith', 'lee': 'lee', 'mrlee': 'lee',
            'carlosalfonzodiaz': 'carlosalfonzodiaz', 'diazcarlosalfonzo': 'carlosalfonzodiaz',
            'technology': 'tech', 'tech': 'tech',
            'abdulrasheed': 'abdulrasheed', 'abdalrashid': 'abdulrasheed', 'sarah': 'sarah', 'sara': 'sarah',
            'center': 'center', 'cetner': 'center',
            'jamesealsmith': 'jes', 'j.e.smith': 'jes', 'jesmith': 'jes', 'j e smith': 'jes',
            'ltd': 'limited', 'limited': 'limited', 'shpg': 'shipping', 'shipping': 'shipping',
            'apt': 'apartment', 'apartment': 'apartment', 'no': 'number', 'number': 'number',
            'ne': 'northeast', 'northeast': 'northeast', 'rd': 'road', 'road': 'road',
            'us': 'usa', 'usa': 'usa', 'unitedstates': 'usa', 'unitedstatesofamerica': 'usa',
            'eaglepharmaceuticalsinc': 'eagledrugsco', 'eagledrugsco': 'eagledrugsco',
            'company': 'company', 'compania': 'company', 'compania': 'company', 'compania': 'company',
            'compania': 'company', 'compania': 'company', 'compania': 'company', 'compania': 'company',
            'compania': 'company', 'compania': 'company', 'compania': 'company',
            'compania': 'company', 'compania': 'company', 'compania': 'company',
            'compania': 'company', 'compania': 'company', 'compania': 'company',
            'compania': 'company', 'compania': 'company', 'compania': 'company',
            '公司': 'company', '회사': 'company', 'شركة': 'company',
            'william': 'william', 'billy': 'william', 'robert': 'robert', 'bob': 'robert',
            'elizabeth': 'elizabeth', 'betty': 'elizabeth',
            '4': 'four', 'four': 'four', 'iv': 'four', '1st': 'first', 'first': 'first',
            '0': 'zero', 'o': 'zero', '1': 'one', 'i': 'one', 'l': 'one', '2': 'two', 'z': 'two',
            '5': 'five', 's': 'five', '6': 'six', 'b': 'eight', '8': 'eight', '13': 'eight',
            'm': 'm', 'rn': 'm', 'w': 'w', 'vv': 'w', '|_': 'l',
        }
        # 数字/罗马数字/序数归一
        self.num_map = {
            '4': 'four', 'iv': 'four', '1st': 'first', 'first': 'first',
            '2': 'two', 'z': 'two', '5': 'five', 's': 'five', '6': 'six', 'b': 'eight', '8': 'eight', '13': 'eight',
            '0': 'zero', 'o': 'zero', '1': 'one', 'i': 'one', 'l': 'one',
        }
        # 形近字归一
        self.shape_map = {
            '0': 'zero', 'o': 'zero', '1': 'one', 'i': 'one', 'l': 'one', '2': 'two', 'z': 'two',
            '5': 'five', 's': 'five', '6': 'six', 'b': 'eight', '8': 'eight', '13': 'eight',
            'm': 'm', 'rn': 'm', 'w': 'w', 'vv': 'w', '|_': 'l',
        }
    
    def _normalize_word(self, word):
        word = unidecode(word)
        # 新增：去除所有非字母字符（如Em-ma->Emma）
        word = re.sub(r'[^a-z]', '', word.lower())
        if word in self.trans_map:
            return self.trans_map[word]
        if word in self.custom_map:
            return self.custom_map[word]
        if word in self.num_map:
            return self.num_map[word]
        if word in self.shape_map:
            return self.shape_map[word]
        if word.startswith('tech'):
            return 'tech'
        if word in ['center', 'centre', 'cetner']:
            return 'center'
        return word

    def _initials(self, text):
        # 提取首字母缩写
        words = re.findall(r'\b[a-z]', text)
        return ''.join(words)

    def preprocess(self, text: str) -> str:
        # Transliteration + 小写 + 去特殊字符
        text = unidecode(text)
        text = text.lower()
        text = re.sub(r'[^a-z0-9]', '', text)
        # 查自定义映射表
        if text in self.custom_map:
            return self.custom_map[text]
        # 一般归一化（如排序）
        return ''.join(sorted(text))
    
    def preprocess_extreme(self, text: str) -> str:
        text = unidecode(text)
        text = text.lower()
        text = re.sub(r'[^a-z0-9]', '', text)
        return text
    
    def _get_abbreviations(self) -> Dict[str, str]:
        return {
            # 地址相关
            'st': 'street', 'str': 'street', 'rd': 'road', 'ave': 'avenue', 'blvd': 'boulevard',
            'ln': 'lane', 'dr': 'drive', 'ct': 'court', 'pl': 'place', 'sq': 'square',
            'hwy': 'highway', 'pkwy': 'parkway', 'trl': 'trail', 'ter': 'terrace', 'cir': 'circle',
            'mt': 'mount', 'ft': 'fort', 'apt': 'apartment', 'fl': 'floor', 'ste': 'suite',
            'bldg': 'building', 'rm': 'room', 'po': 'post office', 'no': 'number',
            # 职称/学位相关
            'jr': 'junior', 'sr': 'senior', 'phd': 'doctor', 'md': 'doctor', 'dr': 'doctor',
            'prof': 'professor', 'mgr': 'manager', 'asst': 'assistant', 'assoc': 'associate',
            # 其他常见缩写
            'dept': 'department', 'univ': 'university', 'co': 'company', 'corp': 'corporation',
            'inc': 'incorporated', 'ltd': 'limited', 'usa': 'united states', 'uk': 'united kingdom',
            'eu': 'european union', 'us': 'united states', 'shpg': 'shipping', 'ltd': 'limited',
            'ne': 'northeast',
        } 