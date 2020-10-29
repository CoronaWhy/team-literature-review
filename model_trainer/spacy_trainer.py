'''
UNDER CONSTRUCTION: Generalized trainer ("updater") for spaCy models that accepts training data and parameters specific to training case

STATUS:  Current code (10/29) is for use with 'models/general_ner/cord_ner' so must be generalized
'''

import os, re, json, pandas as pd, itertools
import spacy, random
from spacy.util import minibatch, compounding
from spacy.gold import GoldParse
from spacy.scorer import Scorer
from statistics import mean

# 9 new COVID entity types
target_entities = ['CORONAVIRUS', 'VIRAL_PROTEIN', 'LIVESTOCK', 'WILDLIFE', 'EVOLUTION', 'PHYSICAL_SCIENCE', 'SUBSTRATE', 'MATERIAL', 'IMMUNE_RESPONSE']
# 14 "Activity" entity types
target_entities = target_entities + ['ACTIVITY', 'BEHAVIOR', 'SOCIAL_BEHAVIOR', 'INDIVIDUAL_BEHAVIOR', 'DAILY_OR_RECREATIONAL_ACTIVITY', 'OCCUPATIONAL_ACTIVITY', 'HEALTH_CARE_ACTIVITY', 'LABORATORY_PROCEDURE', 'DIAGNOSTIC_PROCEDURE', 'THERAPEUTIC_OR_PREVENTIVE_PROCEDURE', 'RESEARCH_ACTIVITY', 'GOVERNMENTAL_OR_REGULATORY_ACTIVITY', 'EDUCATIONAL_ACTIVITY', 'MACHINE_ACTIVITY']
print(target_entities)


############### GET TRAINING DATA ###############

# get character offsets for an entity in a sentence
# takes in sentence tokens and the start/end word offsets
def get_offsets(tokens, start, end):
    assert start < end
    if start != 0:
        start = len(' '.join(tokens[:start])) + 1
    end = len(' '.join(tokens[:end]))
    return start, end

def get_data():
    TRAIN_DATA = [] # store formatted training data here
    json_ner = open('CORD-NER-ner.json')
    json_corpus = open('CORD-NER-corpus.json')

    # only get target entities
    for n,c in zip(json_ner, json_corpus): # for each document
        # load json object
        doc_ner = json.loads(n)
        doc_corpus = json.loads(c)
        assert doc_ner['doc_id'] == doc_corpus['doc_id']
        sents_ner = doc_ner['sents'] # entities for each sent
        sents_corpus = doc_corpus['sents'] # tokens for each sent

        # check for target_entities in each sentence
        for sn,sc in zip(sents_ner, sents_corpus): # for each sentence
            assert sn['sent_id'] == sc['sent_id']
            ents = [x for x in sn['entities'] if x['type'] in target_entities] # sentence entities
            toks = sc['sent_tokens'] # sentence tokens
            # get entities in proper format with offsets, e.g. (14, 19, 'SUBSTRATE')
            ents = [(*get_offsets(toks, x['start'], x['end']), x['type']) for x in ents]

            if ents: # if sentence contains any target_entities
                sent = ' '.join(toks) # sentence string
                sent = re.sub('_', ' ', sent) # replace underscores with spaces
                sent_data = (sent, {'entities':ents})
                TRAIN_DATA.append(sent_data) # add to TRAIN_DATA
    json_ner.close()
    json_corpus.close()

    # save TRAIN_DATA as json
    with open('TRAIN_DATA.json', 'w') as aus:
        json.dump({'DATA':TRAIN_DATA}, aus)
    print('dumped json')

    return TRAIN_DATA


############### TRAIN SPACY MODEL ###############

def train_model(TRAIN_DATA, model=None):
    ITERATIONS = 20
    if model != None: # load existing model
        nlp = spacy.load(model)
    else: # no existing model
        nlp = spacy.blank("en") # create blank model

    # create ner pipeline component
    if 'ner' not in nlp.pipe_names :
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)
    else :
        ner = nlp.get_pipe("ner")

    # add target_entities to labels
    for ent in target_entities:
        ner.add_label(ent)
    print(ner.labels)

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    optimizer = nlp.begin_training()
    with nlp.disable_pipes(*other_pipes): # only train NER
        for itn in range(ITERATIONS):
            print("starting iteration " + str(itn))
            random.shuffle(TRAIN_DATA) # shuffle training data
            loss = {} # record loss
            batches = minibatch(TRAIN_DATA, size=compounding(4., 32., 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=0.35, losses=loss)
            print('loss:', loss['ner'])
            nlp.to_disk('cord_model') # save model
            print('saved model')
    return nlp


############### TEST SPACY MODEL ###############

def predict_entities(nlp, test_text):
    doc = nlp(test_text)
    return [(ent.text, ent.label_) for ent in doc.ents]


#################### EXECUTE ####################

## GET THE TRAINING DATA

TRAIN_DATA = get_data()
print('\n\n\n', TRAIN_DATA)


## TRAIN THE MODEL

with open('TRAIN_DATA.json') as ein: # load TRAIN_DATA
    TRAIN_DATA = json.load(ein)
    TRAIN_DATA = TRAIN_DATA['DATA']

nlp = train_model(TRAIN_DATA) # train model, 20 iterations

print('finished !')
