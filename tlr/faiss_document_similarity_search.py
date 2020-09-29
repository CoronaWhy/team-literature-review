import faiss
import pymongo
import pandas as pd
import numpy as np


READ_ONLY_USER = 'coronawhyguest'
READ_ONLY_PASS = 'coro901na'
DATABASE = 'cord-19'
MONGO_HOST = 'mongodb.coronawhy.org'
VERSION = 'v22'


def get_collection_mongo(host, user, password, database, collection):
    URI = f'mongodb://{user}:{password}@{host}'
    client = pymongo.MongoClient(URI)
    db = client.get_database(database)
    return db[collection]


def get_coronawhy_cord(version='v22'):
    return get_collection_mongo(MONGO_HOST, READ_ONLY_USER, READ_ONLY_PASS, DATABASE, VERSION)


def _get_cord_uids_for_titles(titles):
    """Search an article on a cord collection by it's title and return its cord_uid."""
    collection = get_coronawhy_cord()
    return list(collection.find({'title': {'$in': titles}}, {'cord_uid'}))


def get_uids(titles=None, uids=None):
    result = list()

    if not (titles or uids):
        raise ValueError("Please add search query (title or article id)")

    if titles:
        result.extend([x['cord_uid'] for x in set(_get_cord_uid_for_title(titles))])

    if uids:
        result.extend(uids)

    return result


# While many cord_uid and title examples are listed above,
# the following works only with the first in the list
def get_faiss_artifacts(embeddings, uids):
    """Returns and index and query vector from FAISS Model based on given embeddings.

    Create a matrix to store article embeddings
    Assign dimension for the vector space
    Build the index
    IndexFlatIP: taking inner product of the vectors
    with normalized vectors, the inner product (IP, of IndexFlatIP)
    becomes cosine similarity
    Adding vectors to the index
    Prepare query vector
    """

    xb = np.ascontiguousarray(embeddings).astype(np.float32)
    d = xb.shape[1]
    index = faiss.IndexFlatIP(d)
    faiss.normalize_L2(xb)
    index.add(xb)

    query_vec = np.ascontiguousarray(embeddings.loc[uids]).reshape(1, -1).astype(np.float32)
    faiss.normalize_L2(query_vec)

    return index, query_vec


def document_similarity_search(index, query_vec, embeddings, cord_uids, k):

    D, I = index.search(query_vec, k)
    similar_id_list = I.tolist()[0]
    similar_cord_uid_list = [cid for cid in embeddings.iloc[similar_id_list].index if cid not in cord_uids]
    collection = get_coronawhy_cord()
    return pd.DataFrame(collection.find({'cord_uid': {'$in': similar_cord_uid_list}}))
