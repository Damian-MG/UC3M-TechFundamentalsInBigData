# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 18:45:46 2022

@author: Pablo
"""

__author__      = "DAMIAN Maleno, NEREA Izcue, PABLO Alonso "
__credits__     = ["Damian Maleno", "Nerea Izcue", "Pablo Alonso"]
__version__     = "1.0"
__email__       = ["100484916@alumnos.uc3m.es",]
__status__      = "In dev"

"""
Program to count the matches of a pattern introduced using the keyboard against all the proteins in the dataset using
multiprocessing programming in Python
"""

import itertools
import time
import csv
import re
import matplotlib.pyplot as plt
from operator import itemgetter
import multiprocessing as mp

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
    print("Cpus: "+str(cpus))
    print("Lenght"+str(Lines))
    chunk_size = int(Lines/mp.cpu_count())
    print(chunk_size)
    for i in range (cpus): iterdata[i*chunk_size+1] = (i+1)*chunk_size+1
    print (iterdata)


def chunking_bueno(Lines, File):
    iterdata = {}
    cpus = mp.cpu_count()
    chunk_size = int(Lines/mp.cpu_count())
    for i in range (cpus): iterdata[i*chunk_size] = (i+1)*chunk_size
    return iterdata

def fileToDict(File):
    with open(File) as file:
        reader = csv.reader(open(file))
    return reader
'''
def countMatches(Pattern, File):
    occurrences = {}
    with open(File) as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            occurrences[int(row[0])] = row[1].count(Pattern)
    return occurrences
'''    

def countMatches(File, sequence, chunk_size, position):
    occurrences = {}
    with open(File) as file:
        reader = csv.reader(file)
        #next(reader)
        inf = (position*chunk_size)+1
        print(inf)
        sup = ((position+1)*chunk_size)+1
        for row in itertools.islice(reader,inf,sup):
            occurrences[int(row[0])] = row[1].count(sequence)
    return occurrences

if __name__ == '__main__':
    sequence = getSequence()
    #iterdata = chunking(Lines= lines)
    t_stamp = time.time()
    chunk_size = int(lines/mp.cpu_count())
    pool = mp.Pool(mp.cpu_count())

    results = pool.starmap(countMatches, zip(itertools.repeat(file),
                                              itertools.repeat(str(sequence)),
                                              itertools.repeat(int(chunk_size)), 
                                              [i for i in range(int(mp.cpu_count()))]))

    
    pool.close()    
    pool.join()
    
    result = {}
    for row in results:
        result.update(row)
    
    print("Total Time: {}".format(time.time()-t_stamp))
    hits = {k: v for k,v in sorted(result.items(), key=itemgetter(1), reverse=True)}
    hits_10 =  dict(itertools.islice(hits.items(),10))
    print(hits_10)
    #hits = countMatches(Pattern= str(sequence), File= file)
    #elapsed_time = time.time() - t_stamp
    #print("Elapsed execution time: ",elapsed_time," s")
    #plt.bar(list(hits.keys()), hits.values(), color = 'g')
    #plt.show()
    #{k: v for k,v in sorted(hits.items(), key=itemgetter(1))}