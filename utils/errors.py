"""
Набор исключений, которые могут возникнуть в ходе работы с фреймворком
"""


class DataParseError(Exception):
    def __init__(self, message):
        self.msg = message
