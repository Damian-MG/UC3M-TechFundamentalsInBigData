__author__      = "DAMIAN Maleno, NEREA Izcue, PABLO Alonso "
__credits__     = ["Damian Maleno", "Nerea Izcue", "Pablo Alonso"]
__version__     = "1.0"
__email__       = ["100484916@alumnos.uc3m.es",]
__status__      = "Finished"

"""
Program to get statistical information from the lastest 5 years of women born with
name MARIA in Catalonia and print an histogram
Idescat API:
https://www.idescat.cat/dev/api/?lang=en
"""

from email.policy import default
import requests
import json
import matplotlib.pyplot as plt

# With the documentation get the URL
url = 'https://api.idescat.cat/onomastica/v1/nadons/dades.json?id=40683&class=t&lang=es'
answer = requests.get(url)
items = json.loads(answer.text)
hist = {}
for i in items['onomastica_nadons']['ff']['f']:
    year = str(i['c'])
    babys = int(i['pos1']['v'])
    hist.update({year:babys})
x = list(hist)[-5:]
y = list(hist.values())[-5:]
plt.bar(x,y,color= 'g')
plt.xlabel('Year of birth')
plt.ylabel('Babys (Women) named MARIA')
plt.title('Evolution of babys named MARIA in the last 5 years in Catalonia')
plt.show()