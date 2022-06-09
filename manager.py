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
        self.global_step = 0
    
    def add(self, x):
        self.counter += 1
        self.sum += x
        self.global_step += 1
    
    def avg(self):
        if self.counter == 0:
            raise ValueError(f'counter is {self.counter}')
        return self.sum / self.counter

    def clean(self):
        self.counter = 0 
        self.sum = 0.0

class LogManager():
    def __init__(self, logger_name, logger_path):
        logging.basicConfig(level=logging.INFO)
        
        self.logger=logging.getLogger(logger_name)
        
        formatter = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

        file_handler = logging.FileHandler(logger_path, encoding='UTF-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)

        console = logging.StreamHandler()
        console.setLevel(logging.INFO)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console)

class Manager(AvgManager):
    def __init__(self, file_path):
        super(Manager, self).__init__()
        username = "cdrec"
        password = "ashdui!#@*$7sj"
        client = MongoClient("mongodb://{}:{}@47.243.233.202:8699/CrossDomainRec".format(username, urllib.parse.quote(password)))
        self.db=client.CrossDomainRec
        self.writer=SummaryWriter(file_path)

        make_dir(file_path)
        logger_path = osp.join(file_path, 'run.log')
        self.logger = LogManager('cross_domain', logger_path).logger

        self.info = {}
        self.info['best_recall']=0.0
    
    def record(self, result:dict):
        result["timestamp"] = datetime.datetime.utcnow()
        result["CN_timestamp"] = datetime.datetime.now()
        self.db.CDR_qw.insert_one(result)

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
    
    