"""
WARNING:
ValueError: [E088] Text of length 2795002 exceeds maximum of 1000000.
The parser and NER models require roughly 1GB of temporary memory per 100,000 characters in the input. 
This means long texts may cause memory allocation errors. If you're not using the parser or NER, it's probably safe to increase the `nlp.max_length` limit. 
The limit is in number of characters, so you can check whether your inputs are too long by checking `len(text)`.
"""
import spacy
from spacy.matcher import PhraseMatcher
import re
from spacy.util import filter_spans
import unicodedata
from lxml.html.clean import Cleaner
from SkillsExtractor.utils import mongo_db_handle

spacy_nlp = spacy.load('en_core_web_sm')
db_conn = mongo_db_handle()
db_colls = db_conn['skills_norm_colls']
skills_colls = db_colls.find()
skills_list = list(set([item["normalized"].lower() for item in list(skills_colls)]))

class SkillsExtractorService:
    def __init__(self):
        self.skills = skills_list
        self.nlp = spacy_nlp
        self.nlp.max_length = len(''.join(self.skills)) + 100
        self.matcher = PhraseMatcher(vocab=self.nlp.vocab)
        self.patterns = [self.nlp.make_doc(name) for name in self.skills]
        self.matcher.add("PatternList", self.patterns)
    
    def execute(self, text: str, preprocessing: bool = True):
        text = text.lower()
        if preprocessing:
            text = self.__preprocess_text(text)
        return self.phraseMatcher(text)
    
    def phraseMatcher(self, text):
        doc = self.nlp(text)
        matches = self.matcher(doc)
        span_list = []
        for match_id, start, end in matches:
            span_list.append(doc[start:end])
        filtered = filter_spans(span_list)
        filtered_skills = [x.text for x in filtered]

        final_skill_list = list(set(filtered_skills))
        return final_skill_list
    
    def __preprocess_text(self, text: str):
        regex_map = {
            'html': re.compile(r'<.*?>|<[^>]+>|\s\s+'),
            'whitespaces': re.compile(r' +'), 
            'token_chars': re.compile(r'[^A-Za-z ]')
        }
        lxml_cleaner = Cleaner(javascript=True, style=True )
        return self.__clean_unicode_html(raw_html=lxml_cleaner.clean_html(text), regex_map=regex_map)
    
    def __clean_unicode_html(self, raw_html: str, regex_map: dict):
        clean_html = re.sub(regex_map['html'], '', raw_html)
        clean_html = unicodedata.normalize("NFKD", clean_html)
        clean_html = re.sub(regex_map['token_chars'], ' ', clean_html)
        clean_html = re.sub(regex_map['whitespaces'], ' ', clean_html)
        return clean_html.strip().lower()
