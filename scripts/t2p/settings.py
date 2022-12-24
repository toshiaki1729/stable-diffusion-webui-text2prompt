import os
from modules import scripts

WDLIKE_TAG_PATH = os.path.abspath(os.path.join(scripts.basedir(), 'data/danbooru_wiki_tags.txt'))
WDLIKE_TOKEN_PATH = os.path.abspath(os.path.join(scripts.basedir(), 'data/danbooru_wiki_token_all-mpnet-base-v2.npy'))
WDLIKE_MODEL_NAME = 'sentence-transformers/all-mpnet-base-v2'