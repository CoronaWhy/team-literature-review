import re

#Load models
from models.general_ner import get_sample_type
from models.sample_size_ner import get_sample_size
from models.study_type import get_study_type
from models.time_period_ner import get_age, get_time_period


#Results assembly
def find_target_number(x):

    """Helper function to clean up extracted excerpts and extract the number of days.
        In most cases, it extracts the mean or median value but when range is reported,
        the function extracts the upper bound.
    """
    match = re.match(r'(.*?)(\d+(\.\d+)?)(\s)(day)', x)
    if match and "±" not in x:
        return match.group(2)
    else:
        return re.search(r'\s\d+(\.\d+)?(days)?', x).group()

def assemble_summary_table(table_name:str, metadata, emb):

    """ Assemble a summary table from end to end.
        Params:
        table_name: str, the file name of the target table
        metadata: df, laoded from metadat.csv
        emb: df, loaded from cord_19_embeddings
    """

    # Refreshing target table to include potential additions
    target_columns, final_articles_metadata = refresh_table(table_name, metadata, emb)

    # Retriving full text for data extraction
    df = get_full_text_sentences(final_articles_metadata)

    # Removing sections that can include data from previous/background studies, rather than the study per se
    sections_to_keep = [s for s in df['section'] if s.lower() not in
                    ['title',
                     'abstract',
                     'background',
                     'summary background',
                     'introduction',
                     'discussion',
                     'discussions',
                     'statistical analysis']
                   ]
    df = df[df['section'].isin(sections_to_keep)]

    # Extracting median/mean/average age from full text
    df['Age'] = df['sentence'].apply(lambda x: get_age(x, tpnlp))

    # Extracting sample size
    df_sample_size = df[['cord_uid', 'abstract']].drop_duplicates()
    df_sample_size['Sample Size'] = df_sample_size['abstract'].apply(lambda x: get_sample_size_regex(x) if pd.notnull(x) else x)

    # Extracting type of sample obtained
    df['Sample Obtained'] = df['sentence'].apply(lambda x: get_sample_type(x, cord_ner_nlp))

    # Extracting time period based on context
    if "shedding" in table_name.lower():
        df['extracted'] = df['sentence'].apply(lambda x: get_time_period(x, tpnlp, ['shedding','positive','clearance']))

    if 'incubation' in table_name.lower():
        df['extracted'] = df['sentence'].apply(lambda x: get_time_period(x, tpnlp, ['incubation']))

    # Extracting numeric values from the extracted
    df['Days'] = df['extracted']\
    .apply(lambda x: find_target_number(x) if pd.notnull(x) and re.search(r'\s\d+(\.\d+)?(days)?', x) else x)\
    .apply(lambda x: ' '.join([x, 'days']) if pd.notnull(x) else x)

    df['Range (Days)'] = df['extracted'].apply(lambda x: x[x.find("(")+1:x.find(")")] if (pd.notnull(x)) and (")" in x) else None)

    #### Preparing final component #1: study metadata ####
    output_metadata = final_articles_metadata[['cord_uid','publish_time','title','abstract','url','journal','source_x','Added on']]
    output_metadata['journal'] = output_metadata['journal'].fillna(output_metadata['source_x'])

    # Predicting study type
    test_pool = Pool(
    output_metadata[['abstract', 'title']].fillna(""),
    feature_names=['abstract', 'title'],
    text_features=['abstract', 'title'])

    output_metadata['Study Type'] = get_study_type(test_pool)

    #### Preparing final component #2: extracted metadata ####
    extracted_metadata = df[['cord_uid','Age', 'Sample Obtained']].groupby('cord_uid', as_index=False)\
    .agg(lambda x: list(set(x))) #add sample size, study type

    # Compressing multuple results into one and cleaning up
    extracted_metadata = extracted_metadata.set_index('cord_uid')[['Age','Sample Obtained']]\
    .applymap(lambda x: ', '.join(filter(None,x))).reset_index()

    extracted_metadata['Sample Obtained'] = extracted_metadata['Sample Obtained']\
    .apply(lambda x: x.split(',') if pd.notnull(x) else x)\
    .apply(lambda x: [s.strip() for s in x] if isinstance(x, list) else x)\
    .apply(lambda x: ', '.join(list(set(x))) if isinstance(x, list) else x)

    extracted_metadata = extracted_metadata.merge(df_sample_size[['cord_uid','Sample Size']], on='cord_uid', how='outer')

    #### Combining metadata with findings ####
    output = df[pd.notnull(df['extracted'])][['cord_uid','sentence','extracted','Days', 'Range (Days)']]\
    .merge(output_metadata, on='cord_uid', how='outer')\
    .merge(extracted_metadata,on='cord_uid', how='outer')

    #### Re-arranging and renameing ####
    output=output[['publish_time',
                   'title',
                   'url',
                   'journal',
                   'Study Type',
                   'Sample Size',
                   'Age',
                   'Sample Obtained',
                   'Days',
                   'Range (Days)',
                   'sentence',
                   'Added on']]

    output.columns = ['Date',
                      'Study',
                      'Study Link',
                      'Journal',
                      'Study Type',
                      'Sample Size',
                      'Age',
                      'Sample Obtained',
                      'Days',
                      'Range (Days)',
                      'Excerpt',
                      'Added on']

    # Discarding additions that end up having no data of interest (i.e. false positives)
    output = output[~((output['Added on']==datetime.today().strftime('%-m/%d/%Y')) & (pd.isnull(output['Excerpt'])))]

    return output



table_names = ['Length of viral shedding after illness onset.csv',
               'What is the incubation period of the virus_.csv',
               'Incubation period across different age groups.csv']

for table_name in table_names:
    summary_table = assemble_summary_table(table_name, metadata, emb)
    display(summary_table)
    summary_table.to_csv(table_name, index = False)
