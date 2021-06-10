import os
import json
import logging
from collections import OrderedDict
from datetime import datetime
from operator import itemgetter

import asyncio
from aiokafka import AIOKafkaConsumer
from elasticsearch import AsyncElasticsearch
from elasticsearch.helpers import async_bulk
import numpy as np

from modules.preprocessing import Preprocessor
from modules.clustering import Article, Clusters, Executor


logging.basicConfig(format='%(asctime)s - %(levelname)s -  %(name)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv('LOG_LEVEL', 'INFO').upper())

# Global configuration

KAFKA = {
    'bootstrap_servers': os.getenv('KAFKA_BOOSTRAP_SERVERS', 'localhost:9092'),
    'topic': os.getenv('KAFKA_TOPIC', 'test'),
}
ELASTICSEARCH = {
    'host': os.getenv('ES_HOST', 'localhost:9200'),
    'index': os.getenv('ES_INDEX', 'clustering'),
}
OUTPUT_INTERVAL = int(os.getenv('OUTPUT_INTERVAL', 60*30))  # seconds
CLUSTERING_CONFIG = os.getenv('CLUSTERING_CONFIG', '0.3,0.0,0.7,0.4')

def _handle_task_result(task: asyncio.Task) -> None:
    try:
        task.result()
    except asyncio.CancelledError:
        pass    # Task cancellation should not be logged as an error.
    except Exception as err:
        logger.exception('Exception raised by task "%s"', task.get_name())

async def kafka_handler(queue: asyncio.Queue) -> None:
    """Consume from a Kafka topic and push the messages
    into the queue.

    :param queue: Push kafka messages into this queue
    :type queue: asyncio.Queue
    """
    consumer = AIOKafkaConsumer(
        KAFKA['topic'],
        bootstrap_servers=KAFKA['bootstrap_servers'],
        group_id=None,
        value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        #auto_offset_reset='earliest'
    )

    await consumer.start()

    logger.info('Start kafka handler')

    try:
        # Consume messages
        async for msg in consumer:
            logger.debug("Consume message %d:%d at %o" % (msg.partition, msg.offset, msg.timestamp))
            
            article = msg.value

            # Basic filter for valid articles
            valid_article = ('title' in article) and (article['title']) and len(article['title']) > 10
            valid_article = valid_article and article['lang'] == 'en'
            if not valid_article:
                logger.debug('Skip article %s' % article['submission_id'])
                continue
            
            logger.debug('Put article %s in queue' % article['submission_id'])
            await queue.put(article)            

    finally:
        # Will leave consumer group; perform autocommit if enabled.
        await consumer.stop()
        logger.info('Stop consumer')

async def clustering(queue: asyncio.Queue, executor: Executor) -> None:
    """Implements Mladen's clustering algorithm.
    Articles are pulled from the queue.

    :param queue: Pull articles from this queue
    :type queue: asyncio.Queue
    :param executor: Cluster's executor
    :type executor: Executor
    """
    logger.info('Start clustering algorithm')

    while True:
        article = await queue.get()
        a_id = article['submission_id']
             
        logger.debug('Clustering: process article %s' % a_id)
        
        # Pre-process the article title and text
        entities_title = Preprocessor.get_entities_spacy(article['title'])
        keyphrases_title = Preprocessor.get_keyphrases_pke(article['title'])
        keyphrases_text = Preprocessor.get_keyphrases_pke(article['text'])

        # Add current article to the clusters
        executor.new_article(a_id, (entities_title, keyphrases_title, keyphrases_text))

        # [DEV] Store current article's title
        _add_article_title_2_map(executor, a_id, article['title'])

        queue.task_done()
        logger.debug('Clustering: done for article %s' % a_id)

def _add_article_title_2_map(ex, a_id, title):
    """Store an article's title into the mapping.
    Caps the title length and keep the mappig to a max size
    of 100k.
    """
    # cap to 80 chars
    ex.article_titles[a_id] = title[:80]

    # cap the max num of articles to 100k
    if len(ex.article_titles) > 100000:
        for _ in range(0, 1000):
            ex.article_titles.popitem(last=False)

def _map_article_titles(ex, a_id):
    if a_id in ex.article_titles:
        return ex.article_titles[a_id]
    else:
        return '-'

def get_cluster_distr(ex: Executor, limit: int = 100) -> OrderedDict:
    """Get article titles per each cluster. """
    if ex.clusters.num_clusters() <= 1:
        return OrderedDict()
    
    # get cluster distribution
    distr = ex.clusters.get_cluster_distr()
    
    # get number of articles per cluster (descrease order)
    n = ex.clusters.get_n()
    c_ids = np.argsort(-n)[:limit]

    # for each cluster, retrieve articles titles
    distr_out = OrderedDict()
    for c_id in c_ids:
        distr_out[c_id] = list(map(lambda x: _map_article_titles(ex, x), distr[c_id]))

    return distr_out

def get_top_terms_per_cluster(ex: Executor, k: int = 10, limit: int = 100) -> OrderedDict:
    """Get top k terms per cluster. """
    if ex.clusters.num_clusters() <= 1:
        return OrderedDict()

    _, _, centers = ex.clusters.get_centers()
    _, _, vocab = ex.clusters.get_vocab()

    # Inverted vocabulary for convenience
    vocab_inv = {i: term for term, i in vocab.items()}

    # get number of articles per cluster (descrease order)
    n = ex.clusters.get_n()
    c_ids = np.argsort(-n)[:limit]

    top_terms = OrderedDict()
    for c_id in c_ids:
        t = np.argsort(-centers[c_id].A[0])[:k].tolist()  
        top_terms[c_id] = list(itemgetter(*t)(vocab_inv))

    return top_terms

async def periodic_output(es: AsyncElasticsearch, ex: Executor, interval: int = 300) -> None:
    """Manage the output process periodically (every `interval` seconds).
    NOTE: schedule is subject to time drift over time.

    :param es: Elasticsearch client.
    :type es: AsyncElasticsearch
    :param ex: Cluster's executor
    :type ex: Executor
    :param interval: interval time in seconds, defaults to 600
    :type interval: int, optional
    """
    logger.info('Start periodic-output')
    
    while True:
        await asyncio.sleep(interval)

        logger.info('Periodic-output is now active')
    
        distr = get_cluster_distr(ex)
        top_terms = get_top_terms_per_cluster(ex)

        # print('Clusters distr', distr)
        # print('Clusters top_terms: ', top_terms)

        logger.info('Periodic-output - dump to Elasticsearch')
        await _dump_2_es(es, distr, top_terms)

        logger.info('Periodic-output done')

async def _es_gendata(cluster_distr, top_terms):
    for c_id, titles in cluster_distr.items():
        yield {
            '_index': ELASTICSEARCH['index'],
            'doc': {
                'cluster_id': c_id,
                'articles_titles': titles,
                'top_terms': top_terms[c_id]
            },
        }

async def _dump_2_es(es, cluster_distr=None, top_terms=None):
    if not (cluster_distr and top_terms):
        return

    await es.indices.delete(index=ELASTICSEARCH['index'], ignore=[400, 404])
    
    # dump data
    _, errors = await async_bulk(es, _es_gendata(cluster_distr, top_terms))

    if errors:
        logger.error('ES bulk index error: %s' % errors)

def _get_config(params_str: str) -> dict:
    if not params_str:
        msg = 'Error parsing clustering config %s' % params_str
        logger.fatal(msg)
        raise ValueError(msg)

    a, b, c, thr = map(float, params_str.split(','))
    return dict(a=a, b=b, c=c, thr=thr)

async def main() -> None:
    # Clustering algorithm executor
    config = _get_config(CLUSTERING_CONFIG)
    ex = Executor(config=config)
    logger.info('Clustering config: %s' % config)
    
    # [DEV] Keep a mapping between articles ID and titles
    ex.article_titles = OrderedDict()
    
    # Elasticsearch client
    es = AsyncElasticsearch(ELASTICSEARCH['host'])

    # Articles are moved using this queue
    q = asyncio.Queue()

    # Create producer task
    producer = asyncio.create_task(kafka_handler(queue=q))

    # Create consumers and periodic tasks 
    consumers = list()
    consumers.append(asyncio.create_task(periodic_output(es, ex, interval=OUTPUT_INTERVAL), name='periodic_output'))
    consumers.append(asyncio.create_task(clustering(queue=q, executor=ex), name='clustering'))

    # Register callbacks
    for task in consumers+[producer]:
        task.add_done_callback(_handle_task_result)

    # with both producers and consumers running, wait for the producers to finish
    await producer

    # wait for the remaining tasks to be processed
    await q.join()

    # cancel the consumers, which are now idle
    for c in consumers:
        c.cancel()

    # clear ES
    await es.close()

if __name__ == "__main__":
    logger.info('Start')

    loop = asyncio.get_event_loop()

    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        logger.info('Receive keyboard interrupt')
    finally:
        loop.close()
        
    logger.info('Quit')