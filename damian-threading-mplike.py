__author__      = "DAMIAN Maleno, NEREA Izcue, PABLO Alonso "
__credits__     = ["Damian Maleno", "Nerea Izcue", "Pablo Alonso"]
__version__     = "1.0"
__email__       = ["100484916@alumnos.uc3m.es",]
__status__      = "In dev"

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

# Vamos a tener que hacer dos implementaciones del threading
# La primera cada thread hace como en el multiproceso y se gestiona su propia apertura de datos
# La segunda comparten entre ellas la apertura del fichero, el único problema que podemos encontrar es que 
# haya concurrencia la I/O con el fichero por lo que un thread bloquee al otro

from itertools import islice
from operator import itemgetter
import time
import csv
import re
import matplotlib.pyplot as plt
import threading
import multiprocessing as mp
from collections import ChainMap
import subprocess as sp


file =  "proteins_prueba.csv"

#def getLength(file):
#    return int(sp.getoutput("tail "+file+" -n 1 | tr -dc '0-9'"))

def getSequence():
    check = False
    while check == False:
        sequence = input("Input the pattern to search matching the regex ^[A-D]+$ :\n").upper()
        if re.match(r"^[A-D]+$",sequence):
            check = True
    return(sequence)

def chunking(Lines, Num_threads):
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

def countThread(Start, Stop, Pattern, File, Results):
    #occurrences = {}
    #with open(File) as file:
    #    reader = csv.reader(islice(file,Start,Stop))
    #    for row in reader:
    #        occurrences[int(row[0])] = row[1].count(Pattern)
    #return occurrences
    Results.append([Start, Stop, Pattern, File])
    

if __name__ == '__main__':
    sequence = getSequence()
    # If your threads don't do I/O, synchronization, etc., and there's nothing else running, 1 thread per core will get you the
    # best performance. However that very likely not the case. Adding more threads usually helps, but after some point, they
    # cause some performance degradation.
    num_threads = mp.cpu_count()
    #iterdata = chunking(Lines= getLength(file), Num_threads= num_threads) # para hacer las pruebas de optimización, cambiar num_threads por un entero no mayor que mp.cpu_count()
    results = []
    iterdata = chunking(Lines= 20, Num_threads= num_threads) # para hacer las pruebas de optimización, cambiar num_threads por un entero no mayor que mp.cpu_count()
    iterdata = [(start, stop, sequence, file, results) for start, stop in iterdata.items()]
    ths = []
    t_stamp = time.time()
    for i, thread in enumerate(iterdata):
         th = threading.Thread(name='th%s' %(i+1), target= countThread, args= iterdata[i])   
         th.start()
         ths.append(th)

    for thread in ths:
         th.join()

    print(results)

    print("Elapsed execution time: ",time.time() - t_stamp," s")


    #hits = dict(ChainMap(*results))
    #hits = {k: v for k,v in sorted(hits.items(), key=itemgetter(1), reverse=True)}
    #print({k: hits[k] for k in list(hits)[:5]})
    #hits_10 = dict(islice(hits.items(),10))
    #plt.bar([ str(i) for i in hits_10.keys()], hits_10.values(), color='g')
    #plt.show()