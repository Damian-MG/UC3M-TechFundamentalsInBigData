__author__      = "DAMIAN Maleno, NEREA Izcue, PABLO Alonso "
__credits__     = ["Damian Maleno", "Nerea Izcue", "Pablo Alonso"]
__version__     = "1.0"
__email__       = ["100484916@alumnos.uc3m.es",]
__status__      = "In dev"

"""
Program to get the balance data for the Consumer confidence indicator, that us seasonally adjusted.
And make an histogram out of it
Eurostat API:
https://wikis.ec.europa.eu/display/EUROSTATHELP/API+SDMX+2.1+-+data+discovery
How to build the query:
https://ec.europa.eu/eurostat/web/query-builder/getting-started/query-builder
"""

import requests
import json
import matplotlib as plt

# With query builder get the URL
url = 'http://ec.europa.eu/eurostat/wdds/rest/data/v2.1/json/en/ei_bsco_m?indic=BS-CSMCI&precision=1&unit=BAL&s_adj=SA'
answer = requests.get(url)
values = json.loads(answer.text)
with open('confidence.json', 'w') as json_file:
    json.dump(values, json_file)

# Falta hacer la gráfica pq no entiendo los datos

