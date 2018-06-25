import os
import glob
import csv

# max out file size limit
csv.field_size_limit(999999999)

path = os.getcwd()

data_path = os.path.abspath(os.path.join(os.getcwd(), '../../Data/Twitter-Data/Clean'))
extension = 'csv'


def read_text(filename):
    # text
    text = []
    # for each line, tokenize words
    for line in filename:
        for field in line:
            stripped = field.strip('\"')
            if not (stripped.startswith('@') or stripped.startswith('RT')):
                text.append(field)
    print(text)
    return text


# Open all files in Direction with extension
def open_dir(path, extension):
    corpus = []
    os.chdir(path)
    files = [i for i in glob.glob('*.{}'.format(extension))]
    for file in files:
        print("Opening {}.".format(file))
        f = csv.reader(open(file, 'rU', encoding='latin-1'), delimiter="\n", quotechar='|')
        corpus.append(read_text(f))
    return corpus


corpus = open_dir(data_path, 'csv')

print(corpus)
