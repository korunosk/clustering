import json
import os


MAX_NUM_ARTICLES = 13822

DATA_DIR      = 'data'
LABELS_DIR    = 'labels'
CLUSTERS_DIR  = 'clusters'
DATASET       = [ 'news-please', 'lsir' ][0]


make_data_path      = lambda fname: os.path.join(DATA_DIR, DATASET, fname)
make_labels_path    = lambda fname: os.path.join(DATA_DIR, DATASET, LABELS_DIR, fname)
make_clusters_path  = lambda fname: os.path.join(DATA_DIR, DATASET, CLUSTERS_DIR, fname)


def exist_labels(fname):
    return os.path.exists(make_labels_path(fname))

def load_data(fname):
    with open(make_data_path(fname), mode='r') as fp:
        return json.load(fp)

def load_labels(fname):
    with open(make_labels_path(fname), mode='r') as fp:
        return json.load(fp)

def load_clusters(fname):
    with open(make_clusters_path(fname), mode='r') as fp:
        return json.load(fp)

def dump_data(obj, fname):
    with open(make_data_path(fname), mode='w') as fp:
        json.dump(obj, fp, indent=4)

def dump_labels(obj, fname):
    with open(make_labels_path(fname), mode='w') as fp:
        json.dump(obj, fp, indent=4)

def dump_clusters(obj, fname):
    with open(make_clusters_path(fname), mode='w') as fp:
        json.dump(obj, fp, indent=4)
