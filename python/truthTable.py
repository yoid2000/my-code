import itertools

columns = ['D','G','Z']
numAids = 1

top = '|     | '
sep = '| --- | '
for column in columns:
    top += f" {column} |"
    sep += " --- |"
top += " out |"
sep += " --- |"
if numAids == 1:
    top += " AID |"
    sep += " --- |"
else:
    for i in range(1,numAids+1):
        top += f" AID{i} |"
        sep += " --- |"
print(top)
print(sep)

comb = 0
for thing in itertools.product(range(2), repeat=len(columns)):
    line = f"| C{comb:02} |"
    comb += 1
    for bool in thing:
        line += f" {bool} |"
    line += " X | Y |"
    print(line)