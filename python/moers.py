import pprint
import os.path
import os
import csv
import re
import random
import numpy
import json
import utm
from detect_delimiter import detect

'''
'''

class plates():
    def __init__(self,ccAll):
        all = []
        self.cc2 = []
        self.assigned = {}
        for plate in ccAll:
            cc = plate['abbreviation']
            all.append(cc)
            if len(cc) <= 2:
                self.cc2.append(cc)
        self.cc = {
            'moers':'MO',
            'verynear':['DU'],
            'near':['KK','MH','OB','GEL','BOT','E'],
            'medium':['D','W','WIT','DO','RE','BO'],
            'far':all
        }

    def getPlate(self):
        while True:
            cc = self.getCityCode()
            local = random.choice(self.cc2)
            ran = random.random()
            if ran < 0.4:
                number = random.randint(11,9999)
            elif ran < 0.9:
                number = random.randint(11,999)
            else:
                number = random.randint(11,99)
            plate = f"{cc} {local} {number}"
            if plate in self.assigned:
                continue
            self.assigned[plate] = True
            return plate

    def getCityCode(self):
        ran = random.random()
        if ran < 0.6:
            return self.cc['moers']
        elif ran < 0.7:
            return random.choices(self.cc['verynear'])[0]
        elif ran < 0.8:
            return random.choices(self.cc['near'])[0]
        elif ran < 0.95:
            return random.choices(self.cc['medium'])[0]
        else:
            return random.choices(self.cc['far'])[0]

class person():
    def __init__(self,totalNumber,plt):
        self.totalNumber = totalNumber
        self.persons = {'fliessender':[],'ruhender':[]}
        self.nextIndex = {'fliessender':0,'ruhender':0}
        self.plt = plt
        self.params = {
            'fliessender' : {
                'similarProb': 0.1,
                'jointProb': 0.7,
                'male': 0.7,
                'ages': {'beta':[2,7],'add':15,'mult':100,
                        'uniform':0.1,'range':[18,85]},
                'counts': {'beta':[1,6],'add':0.5,'mult':30,
                        'uniform':0.1,'range':[18,85]},
            },
            'ruhender' : {
                'similarProb': 0.1,
                'jointProb': 0.7,
                'male': 0.45,
                'ages': {'beta':[1.1,1.3],'add':16,'mult':65,
                        'uniform':0.1,'range':[18,85]},
                'counts': {'beta':[1,6],'add':0.5,'mult':30,
                        'uniform':0.1,'range':[18,85]},
            }
        }
        if False:
            self.showHistograms()
        self.buildPops()

    def getPerson(self,fineType):
        if self.nextIndex[fineType] >= len(self.persons[fineType]):
            print(f"ERROR: {fineType}:{self.nextIndex[fineType]}")
        self.nextIndex[fineType] += 1
        return self.persons[fineType][self.nextIndex[fineType]-1]

    def buildPops(self):
        persons = []
        pops = {
            'fliessender': {},
            'ruhender': {}
        }
        for fineType in self.params.keys():
            total = self.totalNumber[fineType] + 1000
            while True:
                gender = self.getGender(fineType)
                age = self.getAge(self.params[fineType]['ages'])
                count = self.getCount(self.params[fineType]['counts'])
                key = str(gender)+':'+str(age)
                if key in pops[fineType]:
                    pops[fineType][key]['counts'].append(count)
                else:
                    pops[fineType][key] = {'gender':gender, 'age':age,
                                           'counts': [count]}
                total -= count
                if total <= 0:
                    break
        for fineType in self.params.keys():
            for key in pops[fineType].keys():
                pops[fineType][key]['counts'].sort()
        if False:
            for fineType in self.params.keys():
                for key in pops[fineType].keys():
                    print(fineType,key,len(pops[fineType][key]['counts']))
                    print(fineType,key,pops[fineType][key]['counts'][:5])
        # We have all the persons, now we want to join the two fine types
        allKeys = list(set([x for x in pops['fliessender'].keys()]+[x for x in pops['ruhender'].keys()]))
        while True:
            if len(allKeys) == 0:
                break
            deleteList = []
            for key in allKeys:
                for fineType1,fineType2 in [ ['fliessender','ruhender'],
                                         ['ruhender','fliessender'] ]:
                    if key not in pops[fineType1]:
                        continue
                    if len(pops[fineType1][key]['counts']) == 0:
                        if (key not in pops[fineType2] or
                           len(pops[fineType2][key]['counts']) == 0):
                            deleteList.append(key)
                        continue
                    doJoint = False
                    doSimilar = False
                    if (random.random() < self.params[fineType1]['jointProb'] and
                        key in pops[fineType2] and
                        len(pops[fineType2][key]['counts'])):
                        doJoint = True
                        if ( random.random() <
                              self.params[fineType1]['similarProb'] and
                              key in pops[fineType2] and
                              len(pops[fineType1][key]['counts']) > 42 and
                              len(pops[fineType2][key]['counts']) > 42 ):
                            doSimilar = True
                    if doJoint and doSimilar:
                        if random.random() < 0.5:
                            index1 = random.randint(0,20)
                            index2 = random.randint(0,20)
                        else:
                            maxIndex1 = len(pops[fineType1][key]['counts'])-1
                            maxIndex2 = len(pops[fineType2][key]['counts'])-1
                            index1 = random.randint(maxIndex1-20,maxIndex1)
                            index2 = random.randint(maxIndex2-20,maxIndex2)
                        persons.append(
                            {'age':pops[fineType1][key]['age'],
                             'gender':pops[fineType1][key]['gender'],
                             fineType1:pops[fineType1][key]['counts'][index1],
                             fineType2:pops[fineType2][key]['counts'][index2]})
                        pops[fineType1][key]['counts'].pop(index1)
                        pops[fineType2][key]['counts'].pop(index2)
                        
                    elif doJoint:
                        maxIndex1 = len(pops[fineType1][key]['counts'])-1
                        maxIndex2 = len(pops[fineType2][key]['counts'])-1
                        index1 = random.randint(0,maxIndex1)
                        index2 = random.randint(0,maxIndex2)
                        persons.append(
                            {'age':pops[fineType1][key]['age'],
                             'gender':pops[fineType1][key]['gender'],
                             fineType1:pops[fineType1][key]['counts'][index1],
                             fineType2:pops[fineType2][key]['counts'][index2]})
                        pops[fineType1][key]['counts'].pop(index1)
                        pops[fineType2][key]['counts'].pop(index2)
                    else:
                        maxIndex1 = len(pops[fineType1][key]['counts'])-1
                        index1 = random.randint(0,maxIndex1)
                        persons.append(
                            {'age':pops[fineType1][key]['age'],
                             'gender':pops[fineType1][key]['gender'],
                             fineType1:pops[fineType1][key]['counts'][index1],
                             fineType2:0})
                        pops[fineType1][key]['counts'].pop(index1)
            #delete here
            for key in deleteList:
                if key in allKeys:
                    index = allKeys.index(key)
                    allKeys.pop(index)
        # ok, now we have all the person histories set.
        # turn them into microdata
        for person in persons:
            pid = ''.join(random.choices('abcdefghijklmonpqustuvwxiz0123456789',k=20))
            number = self.plt.getPlate()
            for fineType in ['ruhender','fliessender']:
                for _ in range(person[fineType]):
                    self.persons[fineType].append({'pid':pid,
                                        'age':person['age'],
                                        'number':number,
                                        'gender':person['gender']})
        for fineType in ['ruhender','fliessender']:
            random.shuffle(self.persons[fineType])
            pp.pprint(self.persons[fineType][:50])
        

    def getGender(self,fineType):
        ran = random.random()
        if ran < self.params[fineType]['male']:
            return 'M'
        else:
            return 'W'

    def getAge(self,ages):
        ran = random.betavariate(ages['beta'][0],ages['beta'][1])
        ran = round((ran * ages['mult']) + ages['add'])
        ran = max(18,ran)
        return ran

    def getCount(self,counts):
        ran = random.betavariate(counts['beta'][0],counts['beta'][1])
        ran = round((ran * counts['mult']) + counts['add'])
        ran = max(1,ran)
        return ran

    def showHistograms(self):
        for fineType in self.params.keys():
            ages = self.params[fineType]['ages']
            test = []
            for _ in range(10000):
                ran = self.getAge(ages)
                test.append(ran)
            hist,edges = numpy.histogram(test,bins=[18,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100])
            print(fineType)
            for i in range(len(hist)):
                print(edges[i],hist[i])
        for fineType in self.params.keys():
            counts = self.params[fineType]['counts']
            test = []
            for _ in range(10000):
                ran = self.getCount(counts)
                test.append(ran)
            hist,edges = numpy.histogram(test,bins=list(range(30)))
            print(fineType)
            for i in range(len(hist)):
                print(edges[i],hist[i])

pp = pprint.PrettyPrinter(indent=4)

fineTypes = ['fliessender','ruhender']
fineNum = {'fliessender':0,'ruhender':0}

homeDir = os.path.join(os.sep,'paul','tables','moers')
#print(homeDir)
ccFile = 'german_city_codes.json'
ccPath = os.path.join(homeDir,ccFile)
ccf = open(ccPath,'r')
ccAll = json.load(ccf)
plt = plates(ccAll)
ccf.close()

allHeaders = {}
allValues = {}
for fineType in fineTypes:
    dir = os.path.join(homeDir,fineType)
    for fname in os.listdir(dir):
        if fname[-3:] != 'csv':
            continue
        fpath = os.path.join(dir,fname)
        f = open(fpath,'r')
        print('----------------------------------')
        print(fpath)
        line = f.readline()
        print(line)
        if ';' in line:
            delimiter = ';'
        else:
            delimiter = ','
        #print(f"delimiter is {delimiter}")
        f.seek(0)
        reader = csv.DictReader(f,delimiter=delimiter)
        for row in reader:
            # This is the first non-header row.
            # We can get the header values from here
            #print(row)
            for header,value in row.items():
                allValues[header] = value
                if header in allHeaders:
                    allHeaders[header] += 1
                else:
                    allHeaders[header] = 1
            break

pp.pprint(allHeaders)
pp.pprint(allValues)

thing = []
for fineType in fineTypes:
    dir = os.path.join(homeDir,fineType)
    for fname in os.listdir(dir):
        if fname[-3:] != 'csv':
            continue
        fpath = os.path.join(dir,fname)
        f = open(fpath,'r')
        line = f.readline()
        if ';' in line:
            delimiter = ';'
        else:
            delimiter = ','
        #print(f"delimiter is {delimiter}")
        f.seek(0)
        reader = csv.DictReader(f,delimiter=delimiter)
        for row in reader:
            row['BussgelderTyp'] = fineType
            fineNum[fineType] += 1
            thing.append(row)

p = person({'fliessender':fineNum['fliessender'],'ruhender':fineNum['ruhender']},plt)

for row in thing:
    if 'Tatort 2' in row:
        row['Tatort2'] = row['Tatort 2']
        del(row['Tatort 2'])
    if 'Tatzeit' in row:
        row['Zeit'] = row['Tatzeit']
        del(row['Tatzeit'])
    if 'Tatzeit/Uhrzeit' in row:
        row['Zeit'] = row['Tatzeit/Uhrzeit']
        del(row['Tatzeit/Uhrzeit'])
    if 'Geldbuße' in row:
        row['Geldbusse'] = row['Geldbuße']
        del(row['Geldbuße'])
    if ' Geldbuße ' in row:
        row['Geldbusse'] = row[' Geldbuße ']
        del(row[' Geldbuße '])
    if '  Geldbuße  ' in row:
        row['Geldbusse'] = row['  Geldbuße  ']
        del(row['  Geldbuße  '])
    if 'lng' not in row:
        row['lng'] = None
    if 'lat' not in row:
        row['lat'] = None
    if 'Tatort' not in row:
        row['Tatort'] = None
    if 'Tatb-Nr.' in row:
        row['TatbNr'] = row['Tatb-Nr.']
        del(row['Tatb-Nr.'])
    else:
        row['TatbNr'] = None
    if 'Tbnr1' not in row:
        row['Tbnr1'] = None
    if '_id' in row:
        del(row['_id'])

allHeaders = {}
allValues = {}
for row in thing:
    for header,value in row.items():
        allValues[header] = value
        if header in allHeaders:
            allHeaders[header] += 1
        else:
            allHeaders[header] = 1
pp.pprint(allHeaders)
pp.pprint(allValues)

# at this point, the headers are properly defined and populated
# now we need to work on the values

# First remove rows with garbage Geldbusse
pre = re.compile(".*[a-zA-Z]+.*")
newThing = []
for row in thing:
    m = pre.match(row['Geldbusse'])
    if m:
        continue
    newThing.append(row)

print(f'len thing is {len(thing)}')
print(f'len newThing is {len(newThing)}')

# Then clean up Geldbusse and turn it into numbers
for row in newThing:
    #print(row['Geldbusse'])
    row['Geldbusse'] = row['Geldbusse'].replace('€','')
    row['Geldbusse'] = row['Geldbusse'].replace(' ','')
    if row['Geldbusse'].count('.') > 1 or row['Geldbusse'].count(',') > 1:
        print(f"ERROR1: {row['Geldbusse']}")
        quit()
    if ',' in row['Geldbusse'] and '.' in row['Geldbusse']:
        row['Geldbusse'] = row['Geldbusse'].replace('0,00','')
    if ',' not in row['Geldbusse'] and '.' not in row['Geldbusse']:
        row['Geldbusse'] += '.0'
    row['Geldbusse'] = row['Geldbusse'].replace(',','.')

    if row['Geldbusse'][-3] == '.' and row['Geldbusse'][-1] == '0':
        row['Geldbusse'] = row['Geldbusse'][:-1]
    row['Geldbusse'] = float(row['Geldbusse'])

# Next get dates all into the same format
for row in newThing:
    if len(row['Tattag']) == 10 and row['Tattag'][2] == '.' and row['Tattag'][5] == '.':
        day,mon,year = row['Tattag'].split('.')
        row['Tattag'] = f'{year}-{mon}-{day}'
    elif len(row['Tattag']) == 8 and row['Tattag'][2] == '.' and row['Tattag'][5] == '.':
        day,mon,year = row['Tattag'].split('.')
        year = '20' + year
        row['Tattag'] = f'{year}-{mon}-{day}'

# and check that we did it alright:
alreadyGood = 0
reversed = 0
smallYear = 0
for row in newThing:
    if len(row['Tattag']) == 10 and row['Tattag'][4] == '-' and row['Tattag'][7] == '-':
        pass
        alreadyGood += 1
    elif len(row['Tattag']) == 10 and row['Tattag'][2] == '.' and row['Tattag'][5] == '.':
        day,mon,year = row['Tattag'].split('.')
        row['Tattag'] = f'{year}-{mon}-{day}'
        reversed += 1
    elif len(row['Tattag']) == 8 and row['Tattag'][2] == '.' and row['Tattag'][5] == '.':
        day,mon,year = row['Tattag'].split('.')
        year = '20' + year
        row['Tattag'] = f'{year}-{mon}-{day}'
        smallYear += 1
    else:
        print(row['Tattag'])
print(f"already good = {alreadyGood} and reversed = {reversed} and small year = {smallYear}")
pp.pprint(fineNum)

# Now add cars to the 'newThing' table
for row in newThing:
    person = p.getPerson(row['BussgelderTyp'])
    row['PID'] = person['pid']
    #row['Alt'] = person['age']
    #row['Geschlecht'] = person['gender']
    row['Nummernschild'] = person['number']
pp.pprint(newThing[:50])

if False:
    check = {}
    for row in newThing:
        pid = row['PID']
        num = row['Nummernschild']
        if pid in check:
            # This will fail is there is more than one number per pid
            check[pid][num] += 1
        else:
            check[pid] = {num:1}

fieldnames = ['PID',
        'Nummernschild',
        'BussgelderTyp',
        'Geldbusse',
        'TatbNr',
        'Tatort',
        'Tatort2',
        'Tattag',
        'Zeit',
        'Tbnr1',
        'lat',
        'lng',
]
csvPath = os.path.join(homeDir,'moers_bussgelder.csv')
with open(csvPath, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in newThing:
        writer.writerow(row)