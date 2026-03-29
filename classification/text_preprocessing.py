# Text pre-processing: 
# 1) Text cleaning - normalisation of case, punctuation, phone names, emojis
# 2) Tokenisation
# 3) Part-of-Speech (POS) Tagging
# 4) 

from pathlib import Path
import nltk
import pandas as pd
import re
import emoji

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.corpus.reader.wordnet import NOUN, VERB, ADJ, ADV

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
# nltk.download('averaged_perceptron_tagger_eng', download_dir=str(NLTK_DATA_DIR))
# nltk.download('universal_tagset', download_dir=str(NLTK_DATA_DIR))

UNPROCESSED_CANDIDATES = PROJ_ROOT / "data/annotation_candidates.csv"

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

BASE_STOPWORDS = stopwords.words('english')
EXCL_STOPWORDS = ["not", "no", "nor", "but", "very", "more", "most", "why", "how"]
RM_EXTRAS = ["'s", "s", "'m", "'re", "'ve", "'ll", "'d"]
STOPWORDS = [word for word in BASE_STOPWORDS if word not in EXCL_STOPWORDS]
STOPWORDS.extend(RM_EXTRAS)

# regex patterns for pos tagging 
PHONE_BRANDS = {"apple", "samsung", "google", "oppo", "vivo", "xiaomi", "android", "iphone", "pixel"}
PHONE_PATTERN = r"^(?:iphone|pixel|xiaomi)_|^galaxy_s\d+"

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

def clean_text(text): 
    '''
    function to clean and normalise text prior to further preprocessing, takes a pandas row and returns the cleaned text
    '''
    # print("\nORIGINAL:", text)
    text = normalise_emojis(text) # emoji normalisation
    text = normalise_case_punctuation(text) # case & punctuation normalisation
    text = normalise_phone_names(text) # phone name normalisation
    text = expand_common_abbreviations(text) # expand common abbreviations or typos
    # print("\nMODIFIED:", text)
    return text

def tokenisation(row, header) -> list: 
    '''
    function to tokenise cleaned text, returns a list of tokens*

    *note - nltk splits some words into more tokens, e.g. don't = ["do", "n't"]
    '''
    text = str(row[header])
    words = word_tokenize(text)
    return words

def pos_tagging(row, header): 
    '''
    function to receive tokenised text, returns a list of POS tagged tokens
    uses universal tagset instead of penn treebank, less specific
    '''
    raw_tokens = row[header]
    pos_tags = pos_tag(raw_tokens, tagset="universal")

    # hardcode to ensure phone names are tagged as noun
    fixed_tags = []
    for token, tag in pos_tags:
        if token in PHONE_BRANDS or re.match(PHONE_PATTERN, token):
            tag = "NOUN"
        fixed_tags.append((token, tag))
    return fixed_tags

def univ_to_wordnet_pos(tag): 
    '''
    function to convert universal tagset to wordnet tagset (for use in lemmatisation function)
    '''
    if tag == "NOUN":
        return NOUN
    if tag == "VERB":
        return VERB
    if tag == "ADJ":
        return ADJ
    if tag == "ADV":
        return ADV
    return NOUN

def lemmatisation(row, header): 
    '''
    function to receive tokenised and POS tagged tokens, and perform lemmatisation on it (for certain types of words)
    parameters: 
        row: tuple(token: str, tag: str)
        header: str
    '''
    wnl = WordNetLemmatizer()
    raw_tokens = row[header]
    lemm_tokens = []
    for token, tag in raw_tokens: 
        # print(f"ORIGINAL: {token}, {tag}")
        if tag in {"NOUN", "VERB", "ADJ", "ADV"}: # only lemmatise certain types of words, to avoid issues like "us" -> "u"
            wordnet_tag = univ_to_wordnet_pos(tag) 
            lemma = wnl.lemmatize(word=token, pos=wordnet_tag)
        else: 
            lemma = token
        lemm_tokens.append((lemma, tag))
        # print(f"MODIFIED: {lemma}, {tag}")
    return lemm_tokens

def remove_stopwords(row, header):
    '''
    function to remove tokens that are stopwords to reduce noise
    '''
    raw_tokens = row[header]
    filtered_tokens = []
    for token, tag in raw_tokens: 
        if token not in STOPWORDS: 
            filtered_tokens.append((token, tag))
    # print(filtered_tokens)
    return filtered_tokens

def get_top_tokens(df, header):
    vocab_dict = {}

    for tokens in df[header]:
        for token, tag in tokens:
            vocab_dict[token] = vocab_dict.get(token, 0) + 1

    sorted_tokens = sorted(vocab_dict.items(), key=lambda item: item[1], reverse=True)
    print(sorted_tokens)
    return sorted_tokens


def main(): 
    df = pd.read_csv(UNPROCESSED_CANDIDATES, usecols=["text"])
    cleaned_header = "cleaned_comments"
    processed_header = "processed_comments"

    # testing run
    sample = df.head(30).copy()
    sample[cleaned_header] = sample["text"].apply(clean_text)
    sample[cleaned_header] = sample.apply(tokenisation, axis=1, args=(cleaned_header,))
    sample[processed_header] = sample.apply(pos_tagging, axis=1, args=(cleaned_header,)) # pos tagging of cleaned and tokenised text
    sample[processed_header] = sample.apply(lemmatisation, axis=1, args=(processed_header,)) # lemmatisation of certain types of words
    sample[processed_header] = sample.apply(remove_stopwords, axis=1, args=(processed_header,)) # removal of selected stopwords
    vocab_dict = get_top_tokens(sample, processed_header)
    sample.to_csv("data/sample_cleaned_preview.csv", index=False)


    # testing run - text cleaning
    # test_mask = df["text"].str.contains(r"s26|pixel|iphone|xiaomi|idk|imo|lol|🔥|😂", case=False, na=False)
    # sample = df.loc[test_mask].head(30).copy()
    # sample[cleaned_header] = sample.apply(clean_text, axis=1)
    # sample.to_csv("data/sample_cleaned_preview.csv", index=False)

    # full run
    # df[cleaned_header] = df.apply(clean_text, axis=1)
    # df.to_csv("data/annotation_candidates_cleaned.csv", index=False)


if __name__ == "__main__":
    main()