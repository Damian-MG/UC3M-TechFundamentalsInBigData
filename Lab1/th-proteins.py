__author__      = "DAMIAN Maleno, NEREA Izcue, PABLO Alonso "
__credits__     = ["Damian Maleno", "Nerea Izcue", "Pablo Alonso"]
__version__     = "1.0"
__email__       = ["100484916@alumnos.uc3m.es","100492040@alumnos.uc3m.es","100483840@alumnos.uc3m.es"]
__status__      = "Finished"

"""
Program to count the matches of a pattern introduced using the keyboard against all the proteins in the dataset using
threading programming in Python
Threading:
- A new thread is spawned within the existing process
- Starting a thread is faster than starting a process
- Memory is shared between al threads
- Mutexes often necessary to control access to shared data
- One GIL (Global Interpreter Lock for all threads)
"""

from itertools import islice
from operator import itemgetter
import time
import csv
import re
import matplotlib.pyplot as plt
import threading
import multiprocessing as mp
from multiprocessing import Lock
from collections import ChainMap
import subprocess as sp

File =  "proteins.csv"

def getLength(file):
    return int(sp.getoutput("tail -n 1 "+file+"| tr -dc '0-9'")))

def getSequence():
    check = False
    while check == False:
        sequence = input("Input the pattern to search matching the regex ^[A-D]+$ :\n").upper()
        if re.match(r"^[A-Z]+$",sequence):
            check = True
    return(sequence)

def chunking(Lines, Num_threads):
    portions = []
    chunk_size = int(Lines/Num_threads)
    for i in range (Num_threads):
        with open(File) as file:
            reader = csv.reader(islice(file,(i*chunk_size+1),(i+1)*chunk_size+1))
            portion = []
            for row in reader:
                portion.append(row)
            portions.append(portion)
    return(portions)

def countThread(Pattern, Rows):
    occurrences = {}
    for row in Rows:
        occurrences[int(row[0])] = row[1].count(Pattern)
    mutex.acquire()
    results.append(occurrences)
    mutex.release()

"""
If your threads don't do I/O, synchronization, etc., and there's nothing else running, 1 thread per core will get you the
best performance. However that very likely not the case. Adding more threads usually helps, but after some point, they
cause some performance degradation.
"""
if __name__ == '__main__':
    sequence = getSequence()
    num_threads = 6 # optimal number of threads with Apple M1 / 4 threads in my intel laptop
    t_stamp = time.time()
    iterdata = chunking(Lines= getLength(File), Num_threads= num_threads)
    results = []
    mutex = Lock()
    iterdata = [(sequence, rows) for rows in iterdata]
    ths = []
    for i, thread in enumerate(iterdata):
         th = threading.Thread(name='th%s' %(i+1), target= countThread, args= iterdata[i])
         th.start()
         ths.append(th)
    for thread in ths:
         th.join()
    print("Elapsed execution time: ",time.time() - t_stamp," s")
    hits = dict(ChainMap(*results))
    hits = {k: v for k,v in sorted(hits.items(), key=itemgetter(1), reverse=True)}
    print({k: hits[k] for k in list(hits)[:5]})
    hits_10 = dict(islice(hits.items(),10))
    plt.bar([ str(i) for i in hits_10.keys()], hits_10.values(), color='g')
    plt.show()
