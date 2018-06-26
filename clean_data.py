import os
import glob
import csv
import regex as re

from nltk.tokenize import RegexpTokenizer, word_tokenize

# max out file size limit
csv.field_size_limit(999999999)

path = os.getcwd()

# define data file and file extension
data_path = os.path.abspath(os.path.join(os.getcwd(), '../../Data/Twitter-Data/Clean'))
extension = 'csv'

# load all brand names from file
brands = open(path + "/Config/brands.txt").read().splitlines()
models = open(path + "/Config/models.txt").read().splitlines()


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
    return model


def pre_process(sentence):
    # define tokenizer
    tokenizer = RegexpTokenizer(r'\w+')
    sentence = sentence.lower()
    sentence = replace_brand(sentence)
    sentence = replace_model(sentence)
    tokens = tokenizer.tokenize(sentence)
    return word_tokenize(" ".join(tokens))


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


# Open all files in Direction with extension
def open_dir(path, extension):
    tokens = []
    os.chdir(path)
    files = [i for i in glob.glob('*.{}'.format(extension))]
    for file in files:
        print("Tokenizing {}.".format(file))
        f = csv.reader(open(file, 'rU', encoding='latin-1'), delimiter="\n", quotechar='|')
        tokens += tokenize_text(f)
    return corpus


corpus = open_dir(data_path, 'csv')

print(corpus)
