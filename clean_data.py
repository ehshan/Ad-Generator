import os
import glob
import csv
import regex as re

from nltk.tokenize import RegexpTokenizer, word_tokenize

# max out file size limit
csv.field_size_limit(999999999)

path = os.getcwd()


# load all brand and model names from file
brands = open(path + "/Config/brands.txt").read().splitlines()
models = open(path + "/Config/models.txt").read().splitlines()


def parse_models(models):
    acronyms = []
    clean_models = []
    for model in models:
        acronyms.append(model) if len(model) <= 3 else clean_models.append(model)
    return acronyms, clean_models


acronyms, models = parse_models(models)


# add hash words to filter to lists
def add_hash(tags):
    temp = []
    for item in tags:
        temp.append(item)
        tag = '#' + item
        temp.append(tag)
    return temp


brands = add_hash(brands)
models = add_hash(models)
acronyms = add_hash(acronyms)


# replace all brand names with BRAND
# added spaces before and after | & BRAND to deal with partial matches
def replace_brand(sentence):
    brand_regex = re.compile('|'.join(map(re.escape, brands)))
    brand = brand_regex.sub(" BRAND ", sentence)
    return brand


# replace all model names with MODEL
# added spaces before and after | & MODEL to deal with partial matches
def replace_model(sentence):
    model_regex = re.compile('|'.join(map(re.escape, models)))
    model = model_regex.sub(" MODEL ", sentence)
    acronym_regex = re.compile(' | '.join(map(re.escape, acronyms)))
    model = acronym_regex.sub(" MODEL ", model)
    return model


# removes any link from corpus % replaces with LINK
def remove_links(sentence):
    sentence = re.sub(r"http\S+", " LINK ", sentence)
    sentence = re.sub(r"pic.twitter.com\S+", "", sentence)
    sentence = re.sub(r"twitter\S+", "", sentence)
    sentence = re.sub(r"com\S+", "", sentence)
    return sentence


# removes all non-ascii character from text
def remove_special(sentence):
    # replace hyphens
    sentence = re.sub(r'â€™', '\'', sentence)
    # all non-ASCII characters with a single space
    sentence = re.sub(r'[^\x00-\x7F]+', ' ', sentence)
    return sentence


# removes any social tags (@, #, etc) from text
def remove_social_tags(sentence):
    # sentence = re.sub(r'@\S*', "", sentence)
    # # sentence = re.sub(r'\b#\w+', "", sentence)
    sentence = re.sub(r"(@\S*|#\S*\S*)", "", sentence)
    return sentence


# applies filter function to sentences
def pre_process(sentence):
    # define tokenizer
    tokenizer = RegexpTokenizer(r'\w+')
    sentence = sentence.lower()
    # apply filtering functions
    sentence = remove_special(sentence)
    sentence = remove_links(sentence)
    sentence = replace_brand(sentence)
    sentence = replace_model(sentence)
    sentence = remove_social_tags(sentence)
    tokens = tokenizer.tokenize(sentence)
    return word_tokenize(" ".join(tokens))


# tokenize each file
def tokenize_text(filename):
    # tokens is a list type
    tokens = []
    # for each line, tokenize words
    for line in filename:
        for field in line:
            stripped = field.strip('\"')
            if not (stripped.startswith('@') or stripped.startswith('RT')):
                cleaned = pre_process(stripped)
                tokens.append(cleaned)
    print(tokens)
    return tokens


# Open all files in Direction with extension and tokenize
def tokenize_dir(path, extension):
    tokens = []
    os.chdir(path)
    files = [i for i in glob.glob('*.{}'.format(extension))]
    for file in files:
        print("Tokenizing {}.".format(file))
        f = csv.reader(open(file, 'rU', encoding='latin-1'), delimiter="\n", quotechar='|')
        tokens += tokenize_text(f)
    return tokens


# clean tokens for language model
def clean_tokens(corpus):
    # remove empty sentences
    raw_sentences = [x for x in corpus if x != []]

    # remove sentences with more than 20 words
    sentences = [x for x in raw_sentences if len(x) <= 20]
    # get the longest sentence in corpus for NN input vector
    max_sentence = max(sentences, key=len)
    max_sentence_len = len(max(sentences, key=len))
    return sentences, max_sentence, max_sentence_len
