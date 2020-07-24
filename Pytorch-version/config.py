import os

root_path = os.path.dirname(os.path.abspath(__file__))

data_path = root_path + '/data/{}/{}/'

text_data = data_path + 'word_data/all'
text_file = data_path + 'word_data/all/author{}'
syntax_file = data_path + 'parse_data/author{}/{}.txt'


blog_data = data_path + '{}/author{}/'
blog_text_file = data_path + '{}/author{}/{}.txt'
blog_syntax_file = data_path + '{}/author{}/{}_syntax.txt'


news_text_file = './data/{}/{}/word_data/{}.csv'
news_syntax_file = './data/{}/{}/parse_data/{}/{}.txt'

saved_path = root_path + '/trained_models/'

datasets = ['imdb', 'blogs', 'CCAT', 'ag_news', 'dbpedia']
