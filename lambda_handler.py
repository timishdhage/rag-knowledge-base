import sys
import pysqlite3

sys.modules["sqlite3"] = pysqlite3

from mangum import Mangum
from src.rag.api import app

handler = Mangum(app)
