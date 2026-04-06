'''
Text Pre-processing v2 - optimisation for DeBERTa / RoBERTa models

Updated pipeline:
    1) Emoji normalisation
    2) Slang / abbreviation expansion
    3) Basic text cleanup (URLs, mentions, quotes, spacing, number commas)
    4) Phone model normalisation

CHANGES FROM V1:
    - Removed tokenisation, POS tagging, lemmatisation, stopword removal
    - Cleaned up normalisation to exclude use of special markers (underscores)
    - Keeps cleaned text as natural language for transformer tokenizers
'''


from pathlib import Path
import pandas as pd
import re
from typing import Optional
from dataclasses import dataclass
from tqdm import tqdm
import logging
from ftfy import fix_text
import unicodedata
import emoji


# logging & progress
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
tqdm.pandas()


# project paths 
PROJ_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJ_ROOT / "data"


# file paths 
UNPROCESSED_CANDIDATES = DATA_DIR / "annotation_candidates.csv"
OUTPUT_PREVIEW = DATA_DIR / "sample_cleaned_preview.csv"
OUTPUT_FULL = DATA_DIR / "annotation_candidates_cleaned.csv"


@dataclass
class PreprocessingConfig: 
    '''
    program config for text preprocessing
    '''
    sample_size: Optional[int] = None # for sample testing, default set to None for full dataset processing (set sample size in main())


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
    ("lmao", "laughing my ass off"),
    ("lol", "laughing out loud"),
    ("nah", "no"),
    ("ngl", "not going to lie"),
    ("omg", "oh my god"),
    ("pics", "pictures"),
    ("pic", "picture"),
    ("pls", "please"),
    ("rn", "right now"),
    ("shouldnt", "should not"),
    ("smh", "shaking my head"),
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
    ("wtf", "what the fuck"),
    ("ya", "yeah"),
]


# pre-compile slang patterns from slang dict 
SLANG_PATTERNS = [(re.compile(rf"\b{re.escape(pattern)}\b", re.IGNORECASE), replacement) for pattern, replacement in SLANG_DICT]


# pre-compiling of regex patterns
MENTION_PATTERN = re.compile(r"@[A-Za-z0-9_.]+", re.IGNORECASE)
URL_PATTERN = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
COMMA_IN_NUMBER = re.compile(r"(?<=\d),(?=\d)", re.IGNORECASE)
MULTI_SPACE = re.compile(r"\s+", re.IGNORECASE)
IPHONE_PATTERN = re.compile(r"\biphone\s*(\d+)(?:\s*(pro\s*max|pro|plus))?\b", re.IGNORECASE)
SAMSUNG_PATTERN = re.compile(r"\b(?:samsung\s*(?:galaxy\s*)?|galaxy\s*)?s(\d+)(?:\s*(ultra|plus|\+))?\b", re.IGNORECASE)
PIXEL_PATTERN = re.compile(r"\b(?:google\s*)?pixel\s*(\d+)(?:\s*(pro))?\b", re.IGNORECASE)
XIAOMI_PATTERN = re.compile(r"\bxiaomi\s*(\d+)(?:\s*(ultra))?\b", re.IGNORECASE)


# pre-compile patterns for zero-width and invisible characters 
INVISIBLE_CHARS_PATTERN = re.compile(r"[\u200B-\u200D\uFEFF]")
SPACE_BEFORE_PUNCT_PATTERN = re.compile(r"\s+([,.!?;:])")


class TextPreprocessor:
    '''
    Preprocessing of YouTube comments for transformer models.
    Performs light cleaning and domain-specific normalisation.
    '''
    def __init__(self, config: PreprocessingConfig = None):
        self.config = config or PreprocessingConfig()
        logger.info(f"Initialized TextPreprocessor with config: {self.config}")
    
    def normalise_emojis(self, text):
        '''
        function to normalise emojis 
        returns string with emojis converted to text (e.g., 🔥 -> emoji_fire)
        '''
        if not text:
            return text
        return emoji.demojize(text, delimiters=("emoji_", " "))

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
    
    def clean_basic_text(self, text):
        '''
        function to normalise punctuation, normalise urls, mentions, invisible unicode artefacts
        '''
        if not text:
            return text
        
        text = fix_text(text) # fix mojibake and other unicode issues
        text = unicodedata.normalize("NFKC", text) # normalise unicode characters to standard form

        text = INVISIBLE_CHARS_PATTERN.sub("", text) # remove zero-width and invisible characters
        text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'") # normalise quotes
        text = MENTION_PATTERN.sub(" @USER ", text) # normalise mentions of other youtube users
        text = URL_PATTERN.sub(" URL ", text) # normalise urls 
        text = COMMA_IN_NUMBER.sub("", text) # preserves numbers with commas as one word 

        text = SPACE_BEFORE_PUNCT_PATTERN.sub(r"\1", text) # remove spaces before punctuation
        text = MULTI_SPACE.sub(" ", text).strip() # cleans up whitespaces
        return text
    
    def normalise_phone_names(self, text):
        '''
        Normalise phone model names to standard format, including iPhone, Samsung Galaxy, Google Pixel, and Xiaomi models.
        '''
        if not text:
            return text
        
        def replace_iphone(match):
            number = match.group(1)
            variant = match.group(2)
            if variant:
                variant = variant.replace(" ", "")
                if variant in {"promax", "pro"}:
                    return f"iPhone{number}Pro"
                elif variant == "plus":
                    return f"iPhone{number}"
            return f"iPhone{number}"
        
        def replace_samsung(match):
            number = match.group(1)
            variant = match.group(2)
            if variant:
                variant = variant.replace(" ", "").replace("+", "")
                if variant == "ultra":
                    return f"GalaxyS{number}Ultra"
                elif variant == "plus":
                    return f"GalaxyS{number}"
            return f"GalaxyS{number}"
        
        def replace_pixel(match):
            number = match.group(1)
            variant = match.group(2)
            if variant:
                return f"Pixel{number}Pro"
            return f"Pixel{number}"
        
        def replace_xiaomi(match):
            number = match.group(1)
            variant = match.group(2)
            if variant:
                return f"Xiaomi{number}Ultra"
            return f"Xiaomi{number}"
        
        text = IPHONE_PATTERN.sub(replace_iphone, text)
        text = SAMSUNG_PATTERN.sub(replace_samsung, text)
        text = PIXEL_PATTERN.sub(replace_pixel, text)
        text = XIAOMI_PATTERN.sub(replace_xiaomi, text)
        return text
    
    def clean_text(self, text):
        '''
        Function to perform cleaning and general normalisation of text:
            1) Emoji normalisation
            2) Slang expansion
            3) Basic text cleanup
            4) Phone name normalisation
        '''
        if pd.isna(text) or not text:
            return ""
        
        text = str(text)
        text = self.normalise_emojis(text) # normalise emojis
        text = self.expand_common_abbreviations(text)  
        text = self.clean_basic_text(text)
        text = self.normalise_phone_names(text)
        
        return text
    
    def process_dataframe(self, df, text_column: str = "text", output_cleaned: str = "cleaned_comments"):
        '''
        function to process dataframe, return new dataframe with cleaned text in new column
        '''
        logger.info(f"Processing {len(df)} rows...")
        df = df.copy()
        logger.info("Cleaning text...")
        df[output_cleaned] = df[text_column].progress_apply(self.clean_text)
        logger.info("Processing complete!")
        return df


def run_text_preprocessing():
    config = PreprocessingConfig(
        sample_size=None  # use None for full dataset, reconfig for testing with different sample sizes
    )

    preprocessor = TextPreprocessor(config) # Set up preprocesser

    logger.info(f"Loading data from {UNPROCESSED_CANDIDATES}...")
    df = pd.read_csv(UNPROCESSED_CANDIDATES, usecols=["comment_id", "text", "comment_category"])
    
    # Sampling dataset for testing
    if config.sample_size:
        logger.info(f"Using sample of {config.sample_size} rows for testing...")
        df = df.head(config.sample_size)
    
    df_processed = preprocessor.process_dataframe(df) # processed df
    
    output_path = OUTPUT_PREVIEW if config.sample_size else OUTPUT_FULL
    try: 
        logger.info(f"Saving processed data to {output_path}...")
        df_processed.to_csv(output_path, index=False)
    except PermissionError:
        # User needs to close file so it can be written to
        logger.error(f"The file {output_path} is currently open. Please close it and run the program again.") 
        return
    except Exception as e:
        logger.error(f"An error occurred while saving the file: {e}")
        return
    logger.info("Done!")

    return df_processed
