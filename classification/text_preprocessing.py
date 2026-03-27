from pathlib import Path
import nltk
import pandas as pd
import re
import emoji

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

PROJ_ROOT = Path(__file__).resolve().parents[1]
NLTK_DATA_DIR = PROJ_ROOT / "nltk_data"
nltk.data.path.append(str(NLTK_DATA_DIR))
# if LookUpError for missing resources raised when running, uncomment the following lines to download required resources just once
# nltk.download("punkt", download_dir=str(NLTK_DATA_DIR))
# nltk.download("punkt_tab", download_dir=str(NLTK_DATA_DIR))
# nltk.download("stopwords", download_dir=str(NLTK_DATA_DIR))
# nltk.download("wordnet", download_dir=str(NLTK_DATA_DIR))
# nltk.download("omw-1.4", download_dir=str(NLTK_DATA_DIR))
# nltk.download("vader_lexicon", download_dir=str(NLTK_DATA_DIR))

UNPROCESSED_CANDIDATES = "data/annotation_candidates.csv"

SLANG_DICT = [
    ("atm", "at the moment"),
    ("bc", "because"),
    ("btw", "by the way"),
    ("cams", "cameras"),
    ("cam", "camera"),
    ("cant", "cannot"),
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
    ("lmao", "laugh"), # expand as laugh or don't change
    ("lol", "laugh"), # expand as laugh or don't change
    ("nah", "no"),
    ("ngl", "not going to lie"),
    ("omg", "oh my god"),
    ("pics", "pictures"),
    ("pic", "picture"),
    ("pls", "please"),
    ("rn", "right now"),
    ("smh", "shaking my head"), # expand as is or change to other meaning
    ("tbh", "to be honest"),
    ("thx", "thanks"),
    ("tho", "though"),
    ("tmrw", "tomorrow"),
    ("u", "you"),
    ("ur", "your"),
    ("wanna", "want to"),
    ("wont", "will not"),
    ("wtf", "what the fuck"), # expand or keep as abbreviation to mean exclamation
    ("ya", "yeah"),
]

def normalise_case_punctuation(text): 
    ''' 
    function to normalise case and punctuation, including the removal of miscellanous information like urls and usernames
    e.g. Hello_World!’ -> hello world
    '''
    text = text.lower()
    text = text.replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"') # normalise quotes
    text = re.sub(r"@[A-Za-z0-9_.]+", " ", text) # remove mentions of other youtube users
    text = re.sub(r"http\S+|www\S+", " ", text) # remove urls 
    text = re.sub(r"(?<=\d),(?=\d)", "", text) # preserves numbers with commas
    text = re.sub(r"(?<=\d)\.(?=\d)", "__decimal_point__", text) # preserves decimals (e.g 1.0 -> 1_decimal_point_0) 
    
    # use if no need punctuation
    # text = re.sub(r"[^a-z0-9_\s']", " ", text) # remove all other special chars, replace with space

    # use if keeping punctuation for emotion
    text = re.sub(r"[^a-z0-9_\s'!?]", " ", text) # removes all other special chars, keeps !?(emotion)
    text = re.sub(r"!{2,}", "!", text) # normalises multiple exclamations
    text = re.sub(r"\?{2,}", "?", text) # normalises multiple question marks

    text = text.replace("__decimal_point__", ".") # restore decimal points (e.g. 1_decimal_point_0 -> 1.0)
    text = re.sub(r"\s+", " ", text).strip() # remove repeated spaces
    return text

def normalise_emojis(text):
    '''
    function to convert emojis to text
    e.g. 🔥 -> emoji_fire 
    '''
    text = emoji.demojize(text, delimiters=(" emoji_", " "))
    return text

def normalise_phone_names(text):
    '''
    function to normalise phone names, accounting for the different variations of brands in our dataset
    '''
    def replace_iphone(phone_match): 
        # normalise iphone according to pro, promax, plus, etc. 
        number = phone_match.group(1)
        variant = phone_match.group(2)
        if variant:
            variant = variant.replace(" ", "")
            if variant in {"pro", "promax"}:
                return f"iphone_{number}_pro"
            if variant == "plus":
                return f"iphone_{number}"

        return f"iphone_{number}"
    text = re.sub(r"\biphone\s*(\d+)(?:\s*(pro\s*max|pro|plus))?\b", replace_iphone, text)
    text = re.sub(r"\b(\d+)(?:\s*(pro\s*max|pro|plus))\b", replace_iphone, text) # normalise iphone according to shortened phrases like 17 pro

    def replace_samsung(phone_match):
        # normalise samsung according to ultra, normal, etc. 
        number = phone_match.group(1)
        variant = phone_match.group(2)
        if variant:
            variant = variant.replace(" ", "").replace("+", "")
            if variant == "ultra":
                return f"galaxy_s{number}_ultra"
            if variant == "plus":
                return f"galaxy_s{number}"
        return f"galaxy_s{number}"
    text = re.sub(r"\b(?:samsung\s*galaxy\s*|galaxy\s*)?s\s*(\d+)(?:\s*(ultra|plus|\+))?\b", replace_samsung, text)

    def replace_pixel(phone_match):
        # normalise pixel according to pro, normal, etc. 
        number = phone_match.group(1)
        variant = phone_match.group(2)
        if variant:
            return f"pixel_{number}_pro"
        return f"pixel_{number}"
    text = re.sub(r"\b(?:google\s*)?pixel\s*(\d+)(?:\s*(pro))?\b", replace_pixel, text)

    def replace_xiaomi(phone_match):
        # normalise xiaomi according to ultra, normal, etc. 
        number = phone_match.group(1)
        variant = phone_match.group(2)
        if variant:
            return f"xiaomi_{number}_ultra"
        return f"xiaomi_{number}"
    text = re.sub(r"\bxiaomi\s*(\d+)(?:\s*(ultra))?\b", replace_xiaomi, text)
    return text

def expand_common_abbreviations(text): 
    '''
    function to expand commmon abbreviations
    e.g. idk -> i don't know
    '''
    for pattern, replacement in SLANG_DICT: 
        pattern = rf"\b{re.escape(pattern)}\b"
        text = re.sub(pattern, replacement, text)
    return text

def clean_comments(row): 
    comment = str(row["text"])
    # print("\nORIGINAL:", comment)
    comment = normalise_emojis(comment) # emoji normalisation
    comment = normalise_case_punctuation(comment) # case & punctuation normalisation
    comment = normalise_phone_names(comment) # phone name normalisation
    comment = expand_common_abbreviations(comment) # expand common abbreviations or typos
    # print("\nMODIFIED:", comment)
    return comment

def main(): 
    df = pd.read_csv(UNPROCESSED_CANDIDATES, usecols=["text"])

    # testing run
    # sample = df.head(10).copy()
    # sample["cleaned_comments"] = sample.apply(clean_comments, axis=1)
    # sample.to_csv("data/sample_cleaned_preview.csv", index=False)
    test_mask = df["text"].str.contains(r"s26|pixel|iphone|xiaomi|idk|imo|lol|🔥|😂", case=False, na=False)
    sample = df.loc[test_mask].head(30).copy()
    sample["cleaned_comments"] = sample.apply(clean_comments, axis=1)
    sample.to_csv("data/sample_cleaned_preview.csv", index=False)

    # full run
    # df["cleaned_comments"] = df.apply(clean_comments, axis=1)
    # df.to_csv("data/annotation_candidates_cleaned.csv", index=False)


if __name__ == "__main__":
    main()