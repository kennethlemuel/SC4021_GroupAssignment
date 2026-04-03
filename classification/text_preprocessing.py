'''
Text pre-processing pipeline: 
    1) Text cleaning - normalisation of case, punctuation, phone names, emojis
    2) Tokenisation
    3) Part-of-Speech (POS) Tagging
    4) Lemmatisation
    5) Stopword removal
'''

from pathlib import Path
import nltk
import pandas as pd
import re
import emoji
from typing import List, Optional
from dataclasses import dataclass
from tqdm import tqdm
import logging

# imports for nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.corpus.reader.wordnet import NOUN, VERB, ADJ, ADV

# logging & progress
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
tqdm.pandas()

# project paths 
PROJ_ROOT = Path(__file__).resolve().parents[1]
NLTK_DATA_DIR = PROJ_ROOT / "nltk_data"
DATA_DIR = PROJ_ROOT / "data"
nltk.data.path.append(str(NLTK_DATA_DIR))

# file paths 
UNPROCESSED_CANDIDATES = DATA_DIR / "annotation_candidates.csv"
OUTPUT_PREVIEW = DATA_DIR / "sample_cleaned_preview.csv"
OUTPUT_FULL = DATA_DIR / "annotation_candidates_cleaned.csv"

@dataclass
class PreprocessingConfig: 
    '''
    program config for text preprocessing
    '''
    remove_stopwords: bool = True
    lemmatise: bool = True
    keep_punctuation: bool = True # to keep ?! for emotions or not 
    exclude_stopwords: List[str] = None
    sample_size: Optional[int] = None

    def __post_init__(self): 
        if self.exclude_stopwords is None: 
            self.exclude_stopwords = ["not", "no", "nor", "but", "very", "more", "most", "why", "how"]

# required resources from nltk
REQUIRED_NLTK_RESOURCES = [
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("corpora/stopwords", "stopwords"),
    ("corpora/wordnet", "wordnet"),
    ("corpora/omw-1.4", "omw-1.4"),
    ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ("taggers/universal_tagset", "universal_tagset"),
]

def ensure_nltk_resources():
    '''
    function to download nltk resources if any are missing
    '''
    for resource_path, download_name in REQUIRED_NLTK_RESOURCES:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            logger.info(f"downloading NLTK resource: {download_name}")
            nltk.download(download_name, download_dir=str(NLTK_DATA_DIR))

# slang dictionary
SLANG_DICT = [
    ("atm", "at the moment"),
    ("bc", "because"),
    ("btw", "by the way"),
    ("cams", "cameras"),
    ("cam", "camera"),
    ("cant", "cannot"),
    ("can't", "cannot"),
    ("coulda", "could have"),
    ("cuz", "because"),
    ("dont", "do not"),
    ("fav", "favorite"),
    ("gonna", "going to"),
    ("gotta", "got to"),
    ("idc", "i do not care"),
    ("idk", "i do not know"),
    ("im", "i am"),
    ("imo", "in my opinion"),
    ("ive", "i have"),
    ("lmao", "laughing my ass off"), # expand as laugh or don't change
    ("lol", "laughing out loud"), # expand as laugh or don't change
    ("nah", "no"),
    ("ngl", "not going to lie"),
    ("omg", "oh my god"),
    ("pics", "pictures"),
    ("pic", "picture"),
    ("pls", "please"),
    ("rn", "right now"),
    ("shouldnt", "should not"),
    ("smh", "shaking my head"), # expand as is or change to other meaning
    ("tbh", "to be honest"),
    ("thx", "thanks"),
    ("tho", "though"),
    ("tmrw", "tomorrow"),
    ("u", "you"),
    ("ur", "your"),
    ("wanna", "want to"),
    ("wont", "will not"),
    ("won't", "will not"),
    ("wouldnt", "would not"),
    ("wtf", "what the fuck"), # expand or keep as abbreviation to mean exclamation
    ("ya", "yeah"),
]

# pre-compile slang patterns from slang dict 
SLANG_PATTERNS = [(re.compile(rf"\b{re.escape(pattern)}\b", re.IGNORECASE), replacement) for pattern, replacement in SLANG_DICT]

# stopword config
BASE_STOPWORDS = set(stopwords.words('english'))
RM_EXTRAS = {"'s", "s", "'m", "'re", "'ve", "'ll", "'d"}

def build_stopwords(exclude):
    '''
    function to build set of stopwords with exclusions
    '''
    exclude = exclude or []
    stopwords_set = BASE_STOPWORDS - set(exclude)
    stopwords_set.update(RM_EXTRAS)
    return stopwords_set

# precompiling of phone regex
PHONE_BRANDS = {"apple", "samsung", "google", "oppo", "vivo", "xiaomi", "android", "iphone", "pixel"}
PHONE_PATTERN = re.compile(r"^(?:iphone|pixel|xiaomi)_|^galaxy_s\d+")

# precompiling of regex patterns
MENTION_PATTERN = re.compile(r"@[A-Za-z0-9_.]+")
URL_PATTERN = re.compile(r"http\S+|www\S+")
COMMA_IN_NUMBER = re.compile(r"(?<=\d),(?=\d)")
DECIMAL_MARKER = "zxqvdecimalzxqv"
DECIMAL_PATTERN = re.compile(r"(?<=\d)\.(?=\d)")
MULTI_EXCLAIM = re.compile(r"!{2,}")
MULTI_QUESTION = re.compile(r"\?{2,}")
MULTI_SPACE = re.compile(r"\s+")
IPHONE_PATTERN = re.compile(r"\biphone\s*(\d+)(?:\s*(pro\s*max|pro|plus))?\b", re.IGNORECASE)
SAMSUNG_PATTERN = re.compile(r"\b(?:samsung\s*galaxy\s*|galaxy\s*)?s\s*(\d+)(?:\s*(ultra|plus|\+))?\b", re.IGNORECASE)
PIXEL_PATTERN = re.compile(r"\b(?:google\s*)?pixel\s*(\d+)(?:\s*(pro))?\b", re.IGNORECASE)
XIAOMI_PATTERN = re.compile(r"\bxiaomi\s*(\d+)(?:\s*(ultra))?\b", re.IGNORECASE)


class TextPreprocessor:
    '''
    preprocessing of youtube comments
    performs cleaning, normalization, tokenization, POS tagging, lemmatization, stopword removal
    '''
    def __init__(self, config: PreprocessingConfig = None):
        self.config = config or PreprocessingConfig()
        self.stopwords = build_stopwords(self.config.exclude_stopwords)
        self.lemmatiser = WordNetLemmatizer()
        
        # ensure relevant nltk data has been downloaded
        ensure_nltk_resources()
        logger.info(f"Initialized TextPreprocessor with config: {self.config}")
    
    def normalise_emojis(self, text):
        '''
        function to normalise emojis 
        returns string with emojis converted to text (e.g., 🔥 -> emoji_fire)
        '''
        if not text:
            return text
        return emoji.demojize(text, delimiters=("emoji_", ""))
    
    def expand_common_abbreviations(self, text: str) -> str:
        '''
        function to expand common slang and abbreviations.   
        important: Must be called BEFORE punctuation removal.
        '''
        if not text:
            return text  
        for pattern, replacement in SLANG_PATTERNS:
            text = pattern.sub(replacement, text)
        return text
    
    def normalise_case_punctuation(self, text):
        '''
        function to normalise case and punctuation, removes urls, mentions
        e.g. Hello_World!’ -> hello world'
        '''
        if not text:
            return text
        
        text = text.lower()
        text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'") # normalise quotes
        text = MENTION_PATTERN.sub(" ", text) # remove mentions of other youtube users
        text = URL_PATTERN.sub(" ", text) # remove urls 
        text = COMMA_IN_NUMBER.sub("", text) # preserves numbers with commas as one word
        text = DECIMAL_PATTERN.sub(DECIMAL_MARKER, text) # preserves decimal points
        
        if self.config.keep_punctuation: # keep !? for emotion, remove other special chars
            text = re.sub(r"[^a-z0-9_\s'!?]", " ", text)
            text = MULTI_EXCLAIM.sub("!", text)
            text = MULTI_QUESTION.sub("?", text)
        else: # remove all special chars except apostrophes
            text = re.sub(r"[^a-z0-9_\s']", " ", text)
        
        text = text.replace(DECIMAL_MARKER, ".") # restores decimal points
        text = MULTI_SPACE.sub(" ", text).strip() # cleans up whitespaces
        return text
    
    def normalise_phone_names(self, text):
        '''
        normalise phone model names to standard format, including iphone, samsung galaxy, google pixel, and xiaomi models.
        '''
        if not text:
            return text
        
        def replace_iphone(match):
            number = match.group(1)
            variant = match.group(2)
            if variant:
                variant = variant.replace(" ", "")
                if variant in {"promax"}:
                    return f"iphone_{number}_pro"
                elif variant == "pro":
                    return f"iphone_{number}_pro"
                elif variant == "plus":
                    return f"iphone_{number}"
            return f"iphone_{number}"
        
        def replace_samsung(match):
            number = match.group(1)
            variant = match.group(2)
            if variant:
                variant = variant.replace(" ", "").replace("+", "")
                if variant == "ultra":
                    return f"galaxy_s{number}_ultra"
                elif variant == "plus":
                    return f"galaxy_s{number}"
            return f"galaxy_s{number}"
        
        def replace_pixel(match):
            number = match.group(1)
            variant = match.group(2)
            if variant:
                return f"pixel_{number}_pro"
            return f"pixel_{number}"
        
        def replace_xiaomi(match):
            number = match.group(1)
            variant = match.group(2)
            if variant:
                return f"xiaomi_{number}_ultra"
            return f"xiaomi_{number}"
        
        text = IPHONE_PATTERN.sub(replace_iphone, text)
        text = SAMSUNG_PATTERN.sub(replace_samsung, text)
        text = PIXEL_PATTERN.sub(replace_pixel, text)
        text = XIAOMI_PATTERN.sub(replace_xiaomi, text)
        return text
    
    def clean_text(self, text):
        '''
        function to perform cleaning and general normalisation of text
        order of operations: 
            1) emoji normalisation
            2) slang expansion (before punctuation removal, in case all punctuation is removed including apostrophes)
            3) case and punctuation normalisation
            4) phone name normalisation
        '''
        if pd.isna(text) or not text:
            return ""
        
        text = str(text)
        text = self.normalise_emojis(text)
        text = self.expand_common_abbreviations(text)  # before punctuation removal
        text = self.normalise_case_punctuation(text)
        text = self.normalise_phone_names(text)
        
        return text
    
    def tokenise(self, text):
        '''
        function to tokenise cleaned text into tokens.
        '''
        if not text:
            return []        
        return word_tokenize(text)

    
    def pos_tag_tokens(self, tokens):
        '''
        function to do POS tagging with custom rules for phone brands.
        '''
        if not tokens:
            return []
        
        pos_tags = pos_tag(tokens, tagset="universal")
        
        # special tags for phone brands 
        fixed_tags = []
        for token, tag in pos_tags:
            if token in PHONE_BRANDS or PHONE_PATTERN.match(token):
                tag = "NOUN"
            fixed_tags.append((token, tag))
        
        return fixed_tags
    
    @staticmethod
    def univ_to_wordnet_pos(tag): 
        '''
        function to convert universal tagset to wordnet tagset (for use in lemmatisation function)
        '''
        if tag == "NOUN":
            return NOUN
        elif tag == "VERB":
            return VERB
        elif tag == "ADJ":
            return ADJ
        elif tag == "ADV":
            return ADV
        return NOUN
    
    def lemmatise_tokens(self, tagged_tokens):
        '''
        function to lemmatise tokens based on their POS tags
        only lemmatises NOUN, VERB, ADJ, ADV to avoid issues like "us" -> "u"
        '''
        if not self.config.lemmatise or not tagged_tokens:
            return tagged_tokens
        
        lemmatised = []
        for token, tag in tagged_tokens:
            if tag in {"NOUN", "VERB", "ADJ", "ADV"}:
                wordnet_tag = self.univ_to_wordnet_pos(tag)
                lemma = self.lemmatiser.lemmatize(token, pos=wordnet_tag)
            else:
                lemma = token
            lemmatised.append((lemma, tag))
        return lemmatised
    
    def remove_stopwords(self, tagged_tokens):
        '''
        function to remove stopwords from tagged tokens
        '''
        if not self.config.remove_stopwords or not tagged_tokens:
            return tagged_tokens
        
        return [(token, tag) for token, tag in tagged_tokens if token not in self.stopwords]
    
    def preprocess(self, text):
        '''
        function to perform full preprocessing, returning as (token, tag)
        '''
        cleaned = self.clean_text(text)
        tokens = self.tokenise(cleaned)
        tagged = self.pos_tag_tokens(tokens)
        lemmatised = self.lemmatise_tokens(tagged)
        filtered = self.remove_stopwords(lemmatised)

        return filtered
    
    def process_dataframe(
        self, 
        df: pd.DataFrame, 
        text_column: str = "text",
        output_cleaned: str = "cleaned_comments",
        output_processed: str = "processed_comments"):
        '''
        function to process entire pandas dataframe 
        args:
            df: input dataframe
            text_column: name of column containing raw text
            output_cleaned: name of cleaned text column
            output_processed: name for processed tokens column
        returns:
            dataframe with columns for cleaned and processed text
        '''
        logger.info(f"Processing {len(df)} rows...")
        df = df.copy()
        logger.info("Cleaning text...") 
        df[output_cleaned] = df[text_column].progress_apply(self.clean_text) # clean text
        logger.info("Tokenising...")
        df[output_cleaned + "_tokens"] = df[output_cleaned].progress_apply(self.tokenise) # tokenise
        logger.info("POS tagging...")
        df[output_processed] = df[output_cleaned + "_tokens"].progress_apply(self.pos_tag_tokens) # POS tagging
        if self.config.lemmatise:
            logger.info("Lemmatising...")
            df[output_processed] = df[output_processed].progress_apply(self.lemmatise_tokens) # lemmatisation (see config)
        if self.config.remove_stopwords:
            logger.info("Removing stopwords...")
            df[output_processed] = df[output_processed].progress_apply(self.remove_stopwords) # remove stopwords (see config)
        df.drop(columns=[output_cleaned + "_tokens"], inplace=True)  # drop intermediate column (only keep output processed)
        logger.info("Processing complete!")
        
        return df


def get_vocabulary_stats(df, token_column):
    '''
    function to get vocab stats for the corpus, returns list of sorted vocab by counts
    '''
    vocab_dict = {}
    
    for tokens in df[token_column]:
        for token, tag in tokens:
            vocab_dict[token] = vocab_dict.get(token, 0) + 1
    
    sorted_tokens = sorted(vocab_dict.items(), key=lambda item: item[1], reverse=True)
    
    return sorted_tokens


def main():
    config = PreprocessingConfig(
        remove_stopwords=True,
        lemmatise=True,
        keep_punctuation=True,
        sample_size=None  # use None for full dataset
    )
    
    preprocessor = TextPreprocessor(config) # set up preprocesser
    
    logger.info(f"Loading data from {UNPROCESSED_CANDIDATES}...")
    df = pd.read_csv(UNPROCESSED_CANDIDATES, usecols=["text", "comment_category"])
    
    # sample of dataset for testing
    if config.sample_size:
        logger.info(f"Using sample of {config.sample_size} rows for testing...")
        df = df.head(config.sample_size)
    
    df_processed = preprocessor.process_dataframe(df) # processed df

    
    # process vocab stats 
    vocab_stats = get_vocabulary_stats(df_processed, "processed_comments")
    logger.info(f"\nTop 20 tokens by frequency:")
    for token, freq in vocab_stats[:20]:
        print(f"  {token}: {freq}")
    
    output_path = OUTPUT_PREVIEW if config.sample_size else OUTPUT_FULL
    df_processed.to_csv(output_path, index=False)
    logger.info("Done!")


if __name__ == "__main__":
    main()
