import spacy
import os

'''
Must cite this paper (https://arxiv.org/abs/2003.12218) with regard to use of this model
Originally gathered from: https://xuanwang91.github.io/2020-03-20-cord19-ner/ (I believe)
'''

def load_model():
    '''
    Note: the current model was trained on spaCy 2.1
    '''
    return spacy.load('cord_ner')


def get_sample_type(sentence, model):

    '''
    Extract SUBSTRATE (sample/specimen) from sentences
    '''

    sentence = str(sentence)
    doc = model(sentence)

    # Getting texts with SUBSTRATE entity label
    ents = [ent.text for ent in doc.ents if ent.label_ == 'SUBSTRATE']
    tokens = sentence.split()

    # We mannually extracted specimens from nasopharygeal swabs as the model wasn't tuned enough to recognize them
    if 'swab' in tokens:
        idx = tokens.index('swab')
        ents.append(' '.join([tokens[idx-1], tokens[idx]]))

    # Checking for texts wtih CORONAVIRUS entity label
    has_corona = 'CORONAVIRUS' in [ent.label_ for ent in doc.ents]

    # Checking mentions of covid-19 in the same sentence increases the confidence that the sentece has relevant information
    if ents and has_corona:
        return ', '.join(ents)
    else:
        return doc
