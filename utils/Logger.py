import logging
import sys

logging.basicConfig(stream=sys.stdout, format='[%(levelname)s] %(message)s')
logger = logging.getLogger('globalLogger')
