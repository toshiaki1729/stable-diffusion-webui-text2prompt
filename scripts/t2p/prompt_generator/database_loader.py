import os
import re
import csv
from typing import Dict
import numpy as np

from .. import settings


class Database:
    def __init__(self, database_path: str, re_filename: re.Pattern[str]):
        self.read_files(database_path, re_filename)       
    

    def read_files(self, database_path: str, re_filename: re.Pattern[str]):
        self.clear()
        self.database_path = database_path
        fn, _ = os.path.splitext(os.path.basename(database_path))
        m = re_filename.match(fn)
        self.size_name = m.group(1)
        self.model_name = m.group(2)
        if self.model_name not in settings.TOKENIZER_NAMES:
            print(f'Cannot use database in {database_path}; Incompatible model name "{self.model_name}"')
            self.clear()
            return
        
        tag_path = os.path.join(os.path.dirname(database_path), f'{self.size_name}_tags.txt')
        if not os.path.isfile(tag_path):
            print(f'Cannot use database in {database_path}; No tag file exists')
            self.clear()
            return
        
        with open(tag_path, mode='r', encoding='utf8', newline='\n') as f:
            self.tags = [l.strip() for l in f.readlines()]
        
        tag_idx_path = os.path.join(os.path.dirname(database_path), f'{self.size_name}_tagidx.csv')
        if not os.path.isfile(tag_idx_path):
            print(f'Cannot read tag indices file. Tag count filter cannot be used.')
        else:
            with open(tag_idx_path, mode='r', encoding='utf8', newline='') as f:
                cr = csv.reader(f)
                for row in cr:
                    self.tag_idx.append((int(row[0]), int(row[1])))
            self.tag_idx.sort(key=lambda t : t[0])
        self.tag_idx = [(0, len(self.tags))] + self.tag_idx
    

    def clear(self):
        self.database_path = ''
        self.model_name = ''
        self.size_name = ''
        self.tag_idx = []
        self.tags = []
        self.data: np.ndarray = None
    

    def ready_to_load(self):
        return self.database_path \
            and self.model_name \
            and self.size_name \
            and self.tags \
            and self.tag_idx
    
    def loaded(self):
        return self.data is not None

    def load(self):
        if not self.ready_to_load(): return None        
        if not self.loaded():
            self.data = np.load(self.database_path)['db']
        return self

    
    def name(self):
        return f'{self.model_name} : {self.size_name}'



class DatabaseLoader:
    def __init__(self, path: str, re_filename: re.Pattern[str]):
        self.datas: Dict[str, Database] = dict()
        self.preload(path, re_filename)
    

    def preload(self, path: str, re_filename: re.Pattern[str]):
        dirs = os.listdir(path)
        for d in dirs:
            filepath = os.path.join(path, d)
            if not os.path.isfile(filepath): continue
            _, ext = os.path.splitext(filepath)
            if ext == '.npz':
                ds = Database(filepath, re_filename)
                self.datas[ds.name()] = ds
        print('[text2prompt] Loaded following databases')
        print(sorted(self.datas.keys()))
    

    def load(self, database_name: str):
        database = self.datas.get(database_name)
        return database.load() if database else None