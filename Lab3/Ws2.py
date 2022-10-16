__author__      = "DAMIAN Maleno, NEREA Izcue, PABLO Alonso "
__credits__     = ["Damian Maleno", "Nerea Izcue", "Pablo Alonso"]
__version__     = "1.0"
__email__       = ["100484916@alumnos.uc3m.es","100492040@alumnos.uc3m.es","100483840@alumnos.uc3m.es"]
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
total_ranking = {}
gender_ranking = {}
baby_number = {}
total_rate = {}
gender_rate = {}
for i in items['onomastica_nadons']['ff']['f']:
    year = int(i['c'])
    t_r = int(i['rank']['total'])
    g_r = int(i['rank']['sex'])
    b_n = int(i['pos1']['v'])
    r_t = float(i['pos1']['w']['total'])
    r_g = float(i['pos1']['w']['sex'])
    total_ranking.update({year:t_r})
    gender_ranking.update({year:g_r})
    baby_number.update({year:b_n})
    total_rate.update({year:r_t})
    gender_rate.update({year:r_g})

# PLOT 1
x = list(total_ranking)[-5:]
y = list(total_ranking.values())[-5:]
plt.bar(x,y,color= 'b')
plt.xlabel('Year of birth')
plt.ylabel('Position in the Ranking')
plt.title('Evolution of the rank position of babys (both genders) named MARIA in the last 5 years in Catalonia')
plt.show()

# PLOT 2
x = list(gender_ranking)[-5:]
y = list(gender_ranking.values())[-5:]
plt.bar(x,y,color= 'g')
plt.xlabel('Year of birth')
plt.ylabel('Position in the Ranking')
plt.title('Evolution of the rank position of babys (just Women) named MARIA in the last 5 years in Catalonia')
plt.show()

# PLOT 3
x = list(baby_number)[-5:]
y = list(baby_number.values())[-5:]
plt.bar(x,y,color= 'r')
plt.xlabel('Year of birth')
plt.ylabel('Babys (Women) named MARIA')
plt.title('Evolution of babys named MARIA in the last 5 years in Catalonia')
plt.show()

# PLOT 4
x = list(total_rate)[-5:]
y = list(total_rate.values())[-5:]
plt.bar(x,y,color= 'm')
plt.xlabel('Year of birth')
plt.ylabel('Rate per 1000 newborn')
plt.title('Evolution of the rate of babys (both genders) named MARIA in the last 5 years in Catalonia')
plt.show()

# PLOT 5
x = list(gender_rate)[-5:]
y = list(gender_rate.values())[-5:]
plt.bar(x,y,color= 'c')
plt.xlabel('Year of birth')
plt.ylabel('Rate per 1000 newborn')
plt.title('Evolution of the rate of babys (just Woman) named MARIA in the last 5 years in Catalonia')
plt.show()
