# ADD START AND END TAGS TO TARGET DATA
# -------------------------------------


# add start and end tags to corpus
def tag_corpus(sentences):
    print("\nTagging corpus...")
    all_sentences = []
    for sentence in sentences:
        sentence[:0] = ['<START>']
        sentence.append('<END>')
        all_sentences.append(sentence)
    return all_sentences


# LABEL CORPUS
# ------------

tags = ['engine', 'body', 'speed', 'elegance', 'safety']


# build a set of labels of each keyword
def tag_builder(word, word_model):
    tags = word_model.most_similar(word)
    labels = [i[0] for i in tags]
    labels.append(word)
    return labels


# return a list with lists of labels for each keyword
def all_tags(list, word_model):
    tags = []
    for word in list:
        tags.append(tag_builder(word, word_model))
    return tags


# will check of any keyword in set are present in a sentence
def check_keywords_present(sentence, all_keywords):
    # convert sentence to set
    sentence_set = set(sentence)
    # each tag list
    for words in all_keywords:
        # check if tag is preset in sentence
        # if so return true
        if bool(sentence_set.intersection(words)):
            return True
    # If nothing in the keyword sets present return false
    return False
