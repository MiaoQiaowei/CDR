from pymongo import MongoClient
from tensorboardX import SummaryWriter
from tools import make_dir

import datetime
import urllib 
import logging
import sys
import os.path as osp


class AvgManager():
    def __init__(self):
        self.counter = 0
        self.sum = 0.
    
    def add(self, x):
        self.counter+=1
        self.sum += x
    
    def avg(self):
        if self.counter == 0:
            raise ValueError(f'counter is {self.counter}')
        return self.sum / self.counter

    def clean(self):
        self.counter = 0 
        self.sum = 0.0

class LogManager():
    def __init__(self, logger_name, logger_path):

        self.logger=logging.getLogger(logger_name)
        
        formatter = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

        file_handler = logging.FileHandler(logger_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)


class Manager(AvgManager):
    def __init__(self, file_path):
        super(Manager, self).__init__()
        username = "cdrec"
        password = "ashdui!#@*$7sj"
        client = MongoClient("mongodb://{}:{}@47.243.233.202:8699/CrossDomainRec".format(username, urllib.parse.quote(password)))
        self.db=client.CrossDomainRec

        make_dir(file_path)
        logger_path = osp.join(file_path, 'run.log')
        self.logger = LogManager('cross_domain', logger_path).logger

        self.writer = SummaryWriter(file_path)
        self.info = {}
    
    def record(self, result:dict):
        result["timestamp"] = datetime.datetime.utcnow()
        result["CN_timestamp"] = datetime.datetime.now()
        self.db.CDR.insert_one(result)


class Tee:
    def __init__(self, fname, mode="a"):
        self.stdout = sys.stdout
        self.file = open(fname, mode)

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()
    
    