import os
import pandas as pd
import csv
import pprint

pp = pprint.PrettyPrinter(indent=4)
raceSuffix = ['A','B','C','D','F','G','H','I']

def includesRace(entry):
    if entry['\ufeffTabId'][-1] in raceSuffix:
        return True

def includesGeo(entry):
    if entry['DataProductType'] == 'Detailed Table':
        return True

def numAttributes(entry):
    tt = entry['TabTitle']
    ttSplit = tt.split(' by ')
    return len(ttSplit)

def attributes(entry):
    tt = entry['TabTitle']
    return(tt.split(' by '))

csvPath = os.path.join('other_data', '2022_DataProductList.csv')
with open(csvPath, mode='r', encoding='utf-8') as infile:
    reader = csv.DictReader(infile)
    tablesData = [row for row in reader]
dimDist = []
att = []
for entry in tablesData:
    att.append(entry['TabTitle'])
    numDim = numAttributes(entry)
    if includesRace(entry):
        numDim += 1
    if includesGeo(entry):
        numDim += 1
    dimDist.append(numDim)
print(f"There are {len(dimDist)} total tables")
for i in range(1,8):
    print(f"There are {dimDist.count(i)} tables with {i} attributes")
#for i in range(len(dimDist)):
    #if dimDist[i] == 6:
        #print(att[i])
