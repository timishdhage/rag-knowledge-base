from mangum import Mangum
from src.rag.api import app

handler = Mangum(app)
