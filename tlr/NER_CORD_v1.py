# download scispaCy pre-traind moded
import sys
import subprocess

subprocess.check_call([sys.executable, '-m','pip', 'inistall', 'https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.3.0/en_ner_jnlpba_md-0.3.0.tar.gz']) #put in requirements?

# install dependencies
import pandas as pd
import numpy as np

import scispacy
import spacy
from spacy import displacy # visualize NER stuff
import en_core_sci_sm # import scispaCy model

# pull in data
# replace user specific file manipulation when we get mongo access
placeholder_text_data = Path("/mnt/c/Users/mngav/Data_Science/Kaggle_COVID-19_Open_Research_Dataset_Challenge_CORD-19/Data/metadata_old.csv")
metadata = pd.read_csv(placeholder_text_data)
text = metadata.loc[0,'abstract'] # working with one sentence for times sake

# instantiate scispaCy model
model = en_ner_jnlpba_md
text_data = text

# NER
# extract and return data with substrate label
def get_substrate(text_data, model):
    """ Extract sample type / specimen from text data """
    
    loaded_model = model.load()
    doc = loaded_model(text_data)
    displacy_image = displacy.render(doc, jupyter = True, style = 'ent')
    return(displacy_image)
    
    

    
    