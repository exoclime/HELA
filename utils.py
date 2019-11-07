
import os
import logging

__all__ = [
    "config_logger"
]


def config_logger(log_file="/dev/null", level=logging.INFO):
    
    class MyFormatter(logging.Formatter):
        
        info_format = "\x1b[32;1m%(asctime)s [%(name)s]\x1b[0m %(message)s"
        error_format = "\x1b[31;1m%(asctime)s [%(name)s] [%(levelname)s]\x1b[0m %(message)s"
        
        def format(self, record):
            
            if record.levelno > logging.INFO:
                self._style._fmt = self.error_format
            else:
                self._style._fmt = self.info_format
            
            res = super(MyFormatter, self).format(record)
            return res
    
    rootLogger = logging.getLogger()
    
    fileHandler = logging.FileHandler(log_file)
    fileFormatter = logging.Formatter("%(asctime)s [%(name)s] [%(levelname)s]> %(message)s")
    fileHandler.setFormatter(fileFormatter)
    rootLogger.addHandler(fileHandler)
    
    consoleHandler = logging.StreamHandler()
    consoleFormatter = MyFormatter()
    consoleHandler.setFormatter(consoleFormatter)
    rootLogger.addHandler(consoleHandler)
    
    rootLogger.setLevel(level)

