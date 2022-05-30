from pymongo import MongoClient
from tensorboardX import SummaryWriter
import datetime
import urllib 
import logging
import sys


class LogManager():
    def __init__(self, logger_name, path):
        logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s -%(module)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S %p',
        filename=path,
        level=logging.INFO)
        self.logger=logging.getLogger(logger_name)
        self.writer = SummaryWriter(path)
        self.step_counter=0

class Manager():
    def __init__(self, file_path):
        username = "cdrec"
        password = "ashdui!#@*$7sj"
        client = MongoClient("mongodb://{}:{}@47.243.233.202:8699/CrossDomainRec".format(username, urllib.parse.quote(password)))
        self.db=client.CrossDomainRec
        self.logger = LogManager('cross_domain', file_path).logger
    
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
    
    