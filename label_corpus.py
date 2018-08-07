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


# label the corpus
def label_corpus(sentences, keywords):
    print("\nLabeling corpus...")
    all_keywords = all_tags(keywords)
    all_labels = []
    # each sentence in the set

    for sentence in sentences:
        # create new list of labels + append the start token
        sentence_label = ['<START>']
        word_order = []
        for keys in all_keywords:
            for key in keys:
                if key in sentence:
                    idx = sentence.index(key)
                    # create a list with the position and the keyword
                    word_order.append([idx, key])
                    # print(word_order)

        ordered_labels = sorted(word_order, key=lambda x: x[0])
        print(ordered_labels)
        for label in ordered_labels:
            sentence_label.append(label[1])
        sentence_label.append('<END>')
        print(sentence_label)
        # append the sentences labels to all list
        all_labels.append(sentence_label)
    return all_labels


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


# label the corpus
def clean_and_label(sentences, keywords):
    print("\nLabeling corpus...")
    all_keywords = all_tags(keywords)
    all_labels = []
    keyword_corpus = []
    # each sentence in the set
    for sentence in sentences:
        if bool(check_keywords_present(sentence, all_keywords)):
            # tag the sentence and add it to the new corpus

            word_order = []
            for keys in all_keywords:
                for key in keys:
                    if key in sentence:
                        idx = sentence.index(key)
                        # create a list with the position and the keyword
                        word_order.append([idx, key])
                        # print(word_order)
            if len(word_order) > 1:
                keyword_corpus.append(sentence)
                ordered_labels = sorted(word_order, key=lambda x: x[0])
                # create new list of labels + append the start token
                sentence_label = ['<START>']
                for label in ordered_labels:
                    sentence_label.append(label[1])
                sentence_label.append('<END>')
                # print(sentence_label)
                # append the sentences labels to all list
                all_labels.append(sentence_label)

    # add start and end tags to keyword corpus
    # keyword_corpus = tag_corpus(keyword_corpus)
    return keyword_corpus, all_labels

