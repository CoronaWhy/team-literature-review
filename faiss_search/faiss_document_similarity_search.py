import pandas as pd
import numpy as np
import os
# !pip install faiss-cpu
import faiss
import pymongo #Connecting to MongoDB


#Define toy data
cord_uid_examples = ['02tnwd4m', '8zchiykl', 'gdsfkw1b', 'byp2eqhd', '7gk8uzo0', '0mtmodmo']

title_examples = ['Nitric oxide: a pro-inflammatory mediator in lung disease?',
                  'The 21st International Symposium on Intensive Care and Emergency Medicine, Brussels, Belgium, 20-23 March 2001',
                  'Protein secretion in Lactococcus lactis: an efficient way to increase the overall heterologous protein production',
                  'Immune pathways and defence mechanisms in honey bees Apis mellifera',
                  'Species-specific evolution of immune receptor tyrosine based activation motif-containing CEACAM1-related immune receptors in the dog',
                  'Novel, Divergent Simian Hemorrhagic Fever Viruses in a Wild Ugandan Red Colobus Monkey Discovered Using Direct Pyrosequencing']

#Embedding filename
emb_filename = 'cord_19_embeddings_2020-07-31.csv'

#Setup link to CoronaWhy's MongoDB
# Read-only credentials to CoronaWhy MongoDB service
mongouser = 'coronawhyguest'
mongopass = 'coro901na'
cord_version = 'v22'
mongo_URI = 'mongodb://cord19-rw:coronaWhy2020@mongodb.coronawhy.org'
client = pymongo.MongoClient(mongo_URI)
db = client.get_database('cord19')
print('Existing collections: ', db.list_collection_names())
collection = db[cord_version]


#Get article IDs from MongoDB
def _get_cord_uid_for_title(study_title):
    cord_uids = list(collection.find({'title': str(study_title)}, {'cord_uid'})) #search by title, return cord_uid
    return cord_uids


#Import embeddings from `~/embeddings/embedding_filename`
def _read_embeddings(embeddings_filename):
    emb_path = '/'.join(('embeddings', embeddings_filename))
    print('emb_path: ', emb_path)
    emb = pd.read_csv(emb_path, header = None, index_col = 0)
    print(emb.head())
    return emb


#While many cord_uid and title examples are listed above, the following works only with the first in the list
def setup_faiss_for_doc_similarity(embedding_filename, title_examples, cord_uid_examples):
    #Import embeddings
    emb = _read_embeddings(embedding_filename)

    #Getting article titles from MongoDB if only given `cord_uid`
    if title_examples and not cord_uid_examples:
        result_cord_uids = list(set([result['cord_uid'] for result in _get_cord_uid_for_title(title_examples[0])]))[0] #filters out `_id` column
        print('Articles with associated `cord_uid`s from MongoDB: ', result_cord_uids)

    elif cord_uid_examples and not title_examples:
        result_cord_uids = list(cord_uid_examples[0])

    else:
        print("Please add search query (title or article id)")
        break

    # Create a matrix to store article embeddings
    xb = np.ascontiguousarray(emb).astype(np.float32)
    # Assign dimension for the vector space
    d = xb.shape[1]

    # Build the index
    index = faiss.IndexFlatIP(d) #IndexFlatIP: taking inner product of the vectors
    print('Index training complete: ', index.is_trained)
    faiss.normalize_L2(xb) #with normalized vectors, the inner product (IP, of IndexFlatIP) becomes cosine similarity
    index.add(xb)# Adding vectors to the index
    print('Total rows in index: ', index.ntotal)

    #Prepare query vector
    query_vec = np.ascontiguousarray(emb.loc[cord_uid_examples[0]]).reshape(1, -1).astype(np.float32)
    faiss.normalize_L2(query_vec)
    print('Query vector normalized!')

    return index, query_vec


#Run actual similarity search
def document_similarity_search(index, query_vec, k, return_cord_uid=False, return_metadata=False):
    similar_id_list=[]
    D, I = index.search(query_vec, k)
    similar_id_list.extend(I.tolist()[0])
    similar_cord_uid_list = [cid for cid in emb.iloc[similar_id_list].index if cid not in result_cord_uids]
    if return_cord_uid:
        print('Articles similar to {}: '.format(result_cord_uids), similar_cord_uid_list, '\n')
    if return_metadata:
        mongo_results = pd.DataFrame(collection.find({'cord_uid': {'$in':similar_cord_uid_list}})) #the last two ('hkrljpn3', 'ocu597fg') are apparently ot present in MongoDB
        return mongo_results

#Run search query
index, query_vec = setup_faiss_for_doc_similarity(embedding_filename, title_examples, cord_uid_examples)
document_similarity_search(index, query_vec, k=6, return_cord_uid=True, return_metadata=True) #the last two ('hkrljpn3', 'ocu597fg') are apparently ot present in MongoDB
