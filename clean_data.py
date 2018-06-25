import os
import glob
import csv

from nltk.tokenize import RegexpTokenizer, word_tokenize

# max out file size limit
csv.field_size_limit(999999999)

path = os.getcwd()

data_path = os.path.abspath(os.path.join(os.getcwd(), '../../Data/Twitter-Data/Clean'))
extension = 'csv'


def pre_process(sentence):
    sentence = sentence.lower()
    tokenizer = RegexpTokenizer(r'\w+')
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
