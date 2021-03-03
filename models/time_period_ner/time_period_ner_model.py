import spacy, re
from itertools import cycle
from word2number import w2n
from dateutil.parser import parse

#Time Period NER Model
# Loading the time period model
tpnlp = spacy.load(os.path.join(local_dir,'time-period-ner-75-v2','TimePeriodNER_75_v2'))

def get_age(sentence, model):

    """Extract median/mean/average age from sentences"""

    sentence = str(sentence)
    # To reduce noise, we extract from senteces reporting median or mean age of the population
    if any(substring in sentence for substring in ['median age','mean age','average age']):
        doc = model(sentence)
        ents = [ent.text for ent in doc.ents]
        if len(ents) > 0:
            return ', '.join(ents)

def get_time_period(sentence: str, model, period_types: list):

    """Extract median/mean/average length of viral shedding time

    params:
    sentence: sentence to extract data from
    model: pretrained/custom spacy model to call
    period_types: list of keywords to indicate shedding, incubation, etc...

    """
    # By switching the
    PERIOD_TYPES = period_types

    sentence = str(sentence)
    # Filtering sentence by context keywords
    if any(s in sentence for s in ['median','mean','average','period','time','duration']) \
    and not any(s in sentence for s in ['diagnosis', 'sampling'])\
    and any(s in sentence for s in PERIOD_TYPES):

        doc = model(sentence)
        context = [ent.text for ent in doc.ents if ent.label_ == 'TPcontext']
        data = [ent.text for ent in doc.ents if ent.label_ == 'TPdata']

        if len(context)*len(data)>0:
            output = [item for item in [c + ': ' + d for c, d in zip(cycle(context), data)] \
                      if any( s in item for s in PERIOD_TYPES) and ('day' in item) and re.search(r'\d+(\.\d+)?', item)]
            if len(output) > 0:
                return  '; '.join(output)
