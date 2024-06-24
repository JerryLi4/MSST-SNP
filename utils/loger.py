import logging

class Logger:
    _instance = None

    def __new__(cls, log_file=None, log_level=logging.DEBUG):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
            cls._instance._init_logger(log_file, log_level)
        return cls._instance

    def _init_logger(self, log_file=None, log_level=logging.DEBUG):
        self.logger = logging.getLogger()
        self.logger.setLevel(log_level)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)

if __name__ == "__main__":

    logger1 = Logger('./my_log.log')
    logger2 = Logger()

    print(logger1 is logger2)
    for i in range(100):
        logger1.info('这是一条信息日志')
        logger2.error('这是一条错误日志')
