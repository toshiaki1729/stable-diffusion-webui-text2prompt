import os
import re
from modules import scripts

def get_abspath(path: str):
    return os.path.abspath(os.path.join(scripts.basedir(), path))


TOKENIZER_NAMES = ['all-mpnet-base-v2', 'all-MiniLM-L6-v2']

TOKENIZER_MODELS = {
    TOKENIZER_NAMES[0]: f'sentence-transformers/{TOKENIZER_NAMES[0]}',
    TOKENIZER_NAMES[1]: f'sentence-transformers/{TOKENIZER_NAMES[1]}'
}

DATABASE_PATH_DANBOORU = get_abspath('data/danbooru')

RE_TOKENFILE_DANBOORU = re.compile(r'(danbooru_[^_]+)_token_([^_]+)')