__author__      = "DAMIAN Maleno, NEREA Izcue, PABLO Alonso "
__credits__     = ["Damian Maleno", "Nerea Izcue", "Pablo Alonso"]
__version__     = "1.0"
__email__       = ["100484916@alumnos.uc3m.es","100492040@alumnos.uc3m.es","100483840@alumnos.uc3m.es"]
__status__      = "Finished"

"""
Program to count the matches of a pattern introduced using the keyboard against all the proteins in the dataset using
sequential programming in Python
"""

import itertools
import time
import csv
import re
import matplotlib.pyplot as plt
from operator import itemgetter

file =  "proteins.csv"

def getSequence():
    check = False
    while check == False:
        sequence = input("Input the pattern to search matching the regex ^[A-D]+$ :\n").upper()
        if re.match(r"^[A-Z]+$",sequence):
            check = True
    return(sequence)

def countMatches(Pattern, File):
    occurrences = {}
    with open(File) as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            occurrences[int(row[0])] = row[1].count(Pattern)
    return occurrences
    
if __name__ == '__main__':
    sequence = getSequence()
    t_stamp = time.time()
    hits = countMatches(Pattern= str(sequence), File= file)
    print("Elapsed execution time: ",time.time() - t_stamp," s")
    hits = {k: v for k,v in sorted(hits.items(), key=itemgetter(1), reverse=True)}
    print({k: hits[k] for k in list(hits)[:5]})
    hits_10 = dict(itertools.islice(hits.items(),10))
    plt.bar([ str(i) for i in hits_10.keys()], hits_10.values(), color='g')
    plt.show()