from clean_data import tokenize_dir
import os

path = os.getcwd()

# define data file and file extension
data_path = os.path.abspath(os.path.join(os.getcwd(), '../../Data/Twitter-Data/Clean'))
extension = 'csv'

corpus = tokenize_dir(data_path, extension)

print(corpus)