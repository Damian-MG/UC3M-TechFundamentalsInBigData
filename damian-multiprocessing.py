__author__      = "DAMIAN Maleno, NEREA Izcue, PABLO Alonso "
__credits__     = ["Damian Maleno", "Nerea Izcue", "Pablo Alonso"]
__version__     = "1.0"
__email__       = ["100484916@alumnos.uc3m.es",]
__status__      = "In dev"

"""
Program to count the matches of a pattern introduced using the keyboard against all the proteins in the dataset using
multiprocessing programming in Python
"""

from itertools import islice
from operator import itemgetter
import time
import csv
import re
import matplotlib.pyplot as plt
import multiprocessing as mp
from collections import ChainMap

file =  "proteins.csv"
lines  = 50000

def getSequence():
    check = False
    while check == False:
        sequence = input("Input the pattern to search matching the regex ^[A-D]+$ :\n").upper()
        if re.match(r"^[A-D]+$",sequence):
            check = True
    return(sequence)

def chunking(Lines):
    iterdata = {}
    cpus = mp.cpu_count()
    chunk_size = int(Lines/mp.cpu_count())
    for i in range (cpus): iterdata[i*chunk_size+1] = (i+1)*chunk_size+1
    return iterdata

def countMatches(Start, Stop, Pattern, File):
    occurrences = {}
    with open(File) as file:
        reader = csv.reader(islice(file,Start,Stop))
        for row in reader:
            occurrences[int(row[0])] = row[1].count(Pattern)
    return occurrences

if __name__ == '__main__':
    sequence = getSequence()
    iterdata = chunking(Lines= lines)
    iterdata = [(start, stop, sequence, file) for start, stop in iterdata.items()]
    t_stamp = time.time()
    pool = mp.Pool(mp.cpu_count())
    results = pool.starmap(countMatches, iterdata)
    pool.close()
    print("Elapsed execution time: ",time.time() - t_stamp," s")
    hits = dict(ChainMap(*results))
    hits = {k: v for k,v in sorted(hits.items(), key=itemgetter(1), reverse=True)}
    print({k: hits[k] for k in list(hits)[:5]})
    hits_10 = dict(islice(hits.items(),10))
    plt.bar([ str(i) for i in hits_10.keys()], hits_10.values(), color='g')
    plt.show()
   
