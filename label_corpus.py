

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
