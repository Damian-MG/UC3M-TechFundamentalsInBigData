import pandas as pd
import matplotlib.pyplot as plt
from timeit import default_timer as timer

pattern=input("Enter the secuence: ")
pattern=pattern.upper() #change to upper case
dataset = pd.read_csv("proteins.csv") # occurrences= [[id, nº occurrences], ...]

def serial_prot(dataset,occurrences):
    for i in range(len(dataset.index)):
        s=dataset.loc[i,'sequence'].count(pattern)
        if s!=0:
            occurrences.append([i+1,s])
    return occurrences        
            
            
        
def maximum(occurrences):
    max=0
    for i in range(len(occurrences)):
        if occurrences[i][1]>max:
            max=occurrences[i][1]
    return max  


def main():
    start = timer()
    occurrences=serial_prot(dataset,[])
    end = timer()
    time = end - start
    print('Elapsed time is : ' + str(time) + 's')
    maxi = maximum(occurrences)
    print(occurrences)#prints all the row numbers where the criteria matches and the amount of times it does
    print("The secuence: " + pattern + " is repeated a maximum of " +str(maxi) +" in the following sequences") 

    occurrences=pd.DataFrame(occurrences, columns = ['id','n_occurrences'])
    n=occurrences.drop(occurrences[occurrences.n_occurrences != maxi].index)
    print(n.to_string(index=False))
    plt.bar(n['id'],n['n_occurrences'], width=6000)
    plt.ylabel('Count')
    plt.xlabel('iD');
    plt.title("Number of occurrences for " + str(pattern) )
    return pattern, occurrences, time,n
if __name__ == "__main__":
    pattern, occurrences, time, n=main()