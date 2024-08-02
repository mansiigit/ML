#Program: For a given set of training data examples stored in a .CSV file, implement anddemonstrate the Candidate-Elimination algorithm to output a description of the set of all hypotheses consistent with the training examples.

import pandas as pd

data = pd.read_csv('enjoysport.csv')

concepts = data.iloc[:, :-1].values
target = data.iloc[:, -1].values

n = len(concepts[0])

specific_h = ['0'] * n
general_h = [['?' for _ in range(n)]]

print("The initialization of the specific and general hypothesis")
print("S0:", specific_h, "\nG0:", general_h)

def learn(concepts, target):
    specific_h = concepts[0].copy()
    general_h = [['?' for _ in range(len(specific_h))] for _ in range(len(specific_h))]

    for i, h in enumerate(concepts):
        if target[i] == "yes":
            print(f"\nThe {i+1} training instance is Positive \n", concepts[i])
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    specific_h[x] = '?'
                    general_h[x][x] = '?'
        else:
            print(f"\nThe {i+1} training instance is Negative \n", concepts[i])
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    general_h[x][x] = specific_h[x]
                else:
                    general_h[x][x] = '?'

        print(f"S{i+1}:\n", specific_h)
        print(f"G{i+1}:\n", general_h)
    
    general_h = [h for h in general_h if h != ['?' for _ in range(len(specific_h))]]
    return specific_h, general_h

s_final, g_final = learn(concepts, target)

print("\nThe Final Specific Hypothesis:")
print(s_final)
print("\nThe Final General Hypothesis:")
print(g_final)
