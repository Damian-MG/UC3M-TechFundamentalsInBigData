from pickle import TRUE
import matplotlib.pyplot as plt
import re
import time
import csv

pattern = input("Please enter the pattern you want to match:").upper()
#pattern = re.findall("[A-D]",pattern)
#print(pattern)

time_stamp = time.time()
l =[]

with open("proteins.csv", 'r') as in_file:
    reader= csv.reader(in_file)
    for i in reader:
        occurrence = i[1].count(pattern)
        if occurrence!=0: l.append((i[0],occurrence))
    l.sort(key= lambda tup: tup[1], reverse=True)
    print(l[:10])
    plt.hist(l[:10])
    plt.show()



    #print(i)
    #mydict = {rows[0]:rows[1] for rows in reader} #aprox 1 segundo, definitivamente paralelizable


print(time_stamp)
