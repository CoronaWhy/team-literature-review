import calendar
import re
# import spacy

#Sample Size NER model
def remove_dates(string):

    """Helper function to remove the dates like Mar 30 2020 or 30 Mar 2020"""

    months = '('+ '|'.join([calendar.month_name[i] for i in range(1,13)]) + ')'
    months_abbr = '('+'|'.join([calendar.month_abbr[i] for i in range(1,13)])+ ')'
    dates = [months + '\s\d{1,2}\s\d{4}',
             '\d{1,2}\s'+months+'\s\d{4}',
             months_abbr + '\s\d{1,2}\s\d{4}',
             '\d{1,2}\s'+months_abbr+'\s\d{4}']

    string = re.sub('|'.join(dates), ' ', string)
    return string

def get_sample_size_regex(abstract:str):

    """Extract sample size from the abstracts"""

    abstract = re.sub(',','',abstract)
    abstract = remove_dates(abstract)
    words_nums = []
    for word in abstract.split():
        try:
            words_nums.append(str(w2n.word_to_num(word)))
        except:
            words_nums.append(word)
    abstract = ' '.join(words_nums)
    if any(w in abstract for w in ['enroll',
                                   'includ',
                                   'review',
                                   'extract',
                                   'divide',
                                   'collect',
                                   'examin',
                                   'evaluat',
                                   'report',
                                   'identif',
                                   'admit',
                                   'of the'])\
    and any(w in abstract for w in ['patients',
                                    'cases',
                                    'men',
                                    'males',
                                    'children',
                                    'articles',
                                    'studies'])\
    and re.search('\s\d+([^\.%]{1,25})(patients|cases|men|males|chidren|articles|studies)', abstract):
        return re.search('\s\d+([^\.%]{1,25})(patients|cases|men|males|chidren|articles|studies)', abstract).group().strip()

# # Code to implement the sample size NER model
# nnlp = spacy.load(os.path.join(local_dir,'sample-size-ner-v3','sentence_level_model_v3'))
# def get_sample_size(abstract, model):
#     """Extract sample size from sentences"""
#     sentences = sent_tokenize(str(abstract).lower())
#     nums=[]
#     for sentence in sentences:
#         doc = model(sentence)
#         ents = [ent.text for ent in doc.ents if ent.label_=='enrolled' or ent.label_=='enrolled_add']
#         try:
#             nums.extend([str(w2n.word_to_num(ent)) for ent in ents if not any(s.isdigit() for s in ent)])
#         except:
#             nums.extend(ents)
#     if len(nums) > 0:
#         return ', '.join(filter(None,nums))
