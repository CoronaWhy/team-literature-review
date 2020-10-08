import configparser
import glob
import json
import os
import shutil
from zipfile import ZipFile

import faiss
import pymongo
import pandas as pd
import numpy as np
from kaggle import KaggleApi


config = configparser.ConfigParser()
config.read('setup.cfg')
BASE_DIR = os.path.dirname(__file__)
DEFAULT_CREDENTIALS = os.path.join(os.path.expanduser('~'), config['kaggle']['credentials'])

DATASET = 'allen-institute-for-ai/CORD-19-research-challenge'
EMBEDDINGS_FILE = 'cord_19_embeddings_*.csv'
DATASET_FOLDER = 'cord_19_embeddings'


def load_credentials(credentials_path):
    with open(credentials_path) as f:
        credentials = json.load(f)

    os.environ['KAGGLE_USERNAME'] = credentials['username']
    os.environ['KAGGLE_KEY'] = credentials['key']


def get_kaggle_client(credentials=DEFAULT_CREDENTIALS):
    load_credentials(credentials)
    api = KaggleApi()
    api.authenticate()
    return api


def get_collection_mongo(host, user, password, database, collection):
    URI = f'mongodb://{user}:{password}@{host}'
    client = pymongo.MongoClient(URI)
    db = client.get_database(database)
    return db[collection]


def get_cord19():
    mongo = config['mongodb']
    return get_collection_mongo(
        mongo['host'], mongo['user'], mongo['password'], mongo['database'], mongo['version'])


def get_uid_from_title(titles):
    """Search an article on a cord collection by it's title and return its cord_uid."""
    collection = get_cord19()
    return list(collection.find({'title': {'$in': titles}}, {'cord_uid'}))


def get_uids(titles=None, uids=None):
    result = list()

    if not (titles or uids):
        raise ValueError("Please add search query (title or article id)")

    if titles:
        result.extend([x['cord_uid'] for x in set(get_uid_from_title(titles))])

    if uids:
        result.extend(uids)

    return result


def faiss_search(embeddings, uids, num_results):
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

    _, matches = index.search(query_vec, num_results)
    similar_embeddings = matches.tolist()[0]

    return [uid for uid in embeddings.iloc[similar_embeddings].index if uid not in uids]


def download_embeddings(file_name=None):

    file_name = file_name or os.path.join(BASE_DIR, 'embeddings.csv')
    zip_path = os.path.join(BASE_DIR, 'embeddings.zip')
    client = get_kaggle_client()
    client.dataset_download_cli(DATASET, EMBEDDINGS_FILE, path=zip_path)

    file = ZipFile(zip_path)

    file.extractall(path='./embeddings')
    file_path = os.path.join(DATASET_FOLDER, EMBEDDINGS_FILE)
    os.rename(os.path.join('embeddings', glob.glob(file_path)), file_name)
    os.remove(zip_path)
    shutil.rmtree('embeddings')


def get_embeddings(embeddings):

    if isinstance(embeddings, (np.ndarray, pd.DataFrame)):
        return embeddings

    if not os.path.isfile(embeddings):
        download_embeddings(embeddings)

    return pd.read_csv(embeddings)


def document_similarity_search(titles=None, uids=None, num_results=5, embeddings=None):

    if titles is None and uids is None:
        raise ValueError('Missing arguments: Either titles or uids should be provided.')

    if embeddings is None:
        BASE_DIR = os.path.dirname(__file__)
        embeddings = os.path.join(BASE_DIR, 'embeddings.csv')

    embeddings = get_embeddings(embeddings)

    cord_uids = get_uids(titles, uids)
    similar_uids = faiss_search(embeddings, cord_uids, num_results)
    collection = get_cord19()
    return pd.DataFrame(collection.find({'cord_uid': {'$in': similar_uids}}))
