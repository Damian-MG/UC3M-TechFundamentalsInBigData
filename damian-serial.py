__author__      = "DAMIAN Maleno, NEREA Izcue, PABLO Alonso "
__credits__     = ["Damian Maleno", "Nerea Izcue", "Pablo Alonso"]
__version__     = "1.0"
__email__       = ["100484916@alumnos.uc3m.es",]
__status__      = "In dev"

"""
Program to count the matches of a pattern introduced using the keyboard against all the proteins in the dataset using
sequential programming in Python
"""

import time
import csv
import re
import matplotlib.pyplot as plt

file =  "proteins.csv"

def getSequence():
    check = False
    while check == False:
        sequence = input("Input the pattern to search matching the regex ^[A-D]+$ :\n").upper()
        if re.match(r"^[A-D]+$",sequence):
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
    elapsed_time = time.time() - t_stamp
    print("Elapsed execution time: ",elapsed_time," s")
    plt.bar(list(hits.keys()), hits.values(), color = 'g')
    plt.show()






