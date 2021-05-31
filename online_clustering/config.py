import json
import os


MAX_NUM_ARTICLES = 10000

DATA_DIR      = 'data'
RES_DIR       = 'res'
CLUSTERS_DIR  = 'clusters'
DATASET       = [ 'lsir', 'news-please' ][1]


make_data_path      = lambda fname: os.path.join(DATA_DIR, DATASET, fname)
make_res_path       = lambda fname: os.path.join(DATA_DIR, DATASET, RES_DIR, fname)
make_clusters_path  = lambda fname: os.path.join(DATA_DIR, DATASET, CLUSTERS_DIR, fname)


def load_data(fname):
    with open(make_data_path(fname), mode='r') as fp:
        return json.load(fp)

def load_res(fname):
    with open(make_res_path(fname), mode='r') as fp:
        return json.load(fp)

def load_clusters(fname):
    with open(make_clusters_path(fname), mode='r') as fp:
        return json.load(fp)

def dump_data(obj, fname):
    with open(make_data_path(fname), mode='w') as fp:
        json.dump(obj, fp, indent=4)

def dump_res(obj, fname):
    with open(make_res_path(fname), mode='w') as fp:
        json.dump(obj, fp, indent=4)

def dump_clusters(obj, fname):
    with open(make_clusters_path(fname), mode='w') as fp:
        json.dump(obj, fp, indent=4)
