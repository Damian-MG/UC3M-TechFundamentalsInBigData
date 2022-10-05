__author__      = "DAMIAN Maleno, NEREA Izcue, PABLO Alonso "
__credits__     = ["Damian Maleno", "Nerea Izcue", "Pablo Alonso"]
__version__     = "1.0"
__email__       = ["100484916@alumnos.uc3m.es",]
__status__      = "In dev"

"""
Program to get statistical information from the lastest 5 years of women born with
name MARIA in Catalonia and print an histogram
Idescat API:
https://wikis.ec.europa.eu/display/EUROSTATHELP/API+SDMX+2.1+-+data+discovery
How to build the query:
https://www.idescat.cat/dev/api/?lang=en
https://api.idescat.cat/onomastica/v1/nadons/dades.{…}
"""

import requests
import json
import matplotlib as plt

# With query builder get the URL
url = 'https://api.idescat.cat/onomastica/v1/nadons/dades.json?id=40683&lang=en'

answer = requests.get(url)
values = json.loads(answer.text)
with open('nadons.json', 'w') as json_file:
    json.dump(values, json_file)

# Falta hacer la gráfica pq no entiendo los datos