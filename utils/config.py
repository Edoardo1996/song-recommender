"""
Environment configuration module
"""
from os import environ, path
from dotenv import load_dotenv

basedir = path.abspath(path.dirname(__file__))
load_dotenv(path.join(basedir, '.env'))

for el in environ:
    globals()[f"{el}"] = environ.get(el)
