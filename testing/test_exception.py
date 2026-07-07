import sys
from src.exception import CustomException
from src.logger import logging

try:
    logging.info("Exception test started.")
    a = 10
    b = 0
    result = a / b

except Exception as e:
    logging.error(e)
    print(CustomException(e, sys))