__author__      = "DAMIAN Maleno, NEREA Izcue, PABLO Alonso "
__credits__     = ["Damian Maleno", "Nerea Izcue", "Pablo Alonso"]
__version__     = "1.0"
__email__       = ["100484916@alumnos.uc3m.es",]
__status__      = "In dev"

"""
Program to get the balance data for the Consumer Confidence Indicator, that us seasonally adjusted.
And make an histogram out of it
Eurostat API:
https://wikis.ec.europa.eu/display/EUROSTATHELP/API+SDMX+2.1+-+data+discovery
How to build the query:
https://ec.europa.eu/eurostat/web/query-builder/getting-started/query-builder
"""

import requests
import json
import matplotlib.pyplot as plt

# With query builder get the URL
url = 'http://ec.europa.eu/eurostat/wdds/rest/data/v2.1/json/en/ei_bsco_m?indic=BS-CSMCI&precision=1&unit=BAL&s_adj=SA'
answer = requests.get(url)
answer = json.loads(answer.text)
# Country index
country_index = int(answer['dimension']['geo']['category']['index']['EU27_2020'])
# Last 12 month nanmes and indexes
month_names = answer['dimension']['time']['category']['index']
month_names = list(month_names)[-12:]
month_indexes = answer['dimension']['time']['category']['index'].values()
month_indexes = list(month_indexes)[-12:]
# Observations per country
observations = int(answer['size'][4])
# Consumer confidence indicator
cc_indicator = []
for month in month_indexes:
    cc_indicator.append(float(answer['value'][str(observations*country_index+int(month))]))
# Barplot
plt.bar(month_names,cc_indicator,color= 'c')
plt.title('Evolution of the monthly consumer confidence in the last 12 months')
plt.xlabel('Month')
plt.ylabel('Consumer Confidence Indicator')
plt.show()


