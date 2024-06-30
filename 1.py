import csv

hypo = []
data = []

with open('enjoysport.csv') as csv_file:
    fd = csv.reader(csv_file)
    
    
    print("\nThe given training examples are:")
    for line in fd:
        print(line)
        
        
        if line[-1] == "Yes":
            data.append(line)
    
    
    print("\nThe positive examples are:")
    for x in data:
        print(x)
    
   
    row = len(data)
    col = len(data[0]) if data else 0
    
    
    hypo = data[0][:col-1] 
    
    
    for i in range(1, row):
        for j in range(col-1): 
            if hypo[j] != data[i][j]:
                hypo[j] = '?'
    
    
    print("\nThe maximally specific Find-S hypothesis for the given training examples is:")
    print(hypo)
