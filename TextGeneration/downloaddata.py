# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 08:35:48 2018

@author: santhob
"""

import requests

url = 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt'
r = requests.get(url, allow_redirects=True)
open('gabc.ico', 'wb').write(r.content)