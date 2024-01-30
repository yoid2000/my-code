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

class features():
    def __init__(self,path):
        self.numFoundKey = 0
        self.numNoKey = 0
        self.numFullMatch = 0
        self.numKeyAndNumber = 0
        self.numNumber = 0
        self.bestEffort = 0
        f = open(path,'r',encoding='utf-8')
        features = json.load(f)
        f.close()
        self.streets = {}
        for thing in features['features']:
            street = thing['properties']['STRASSENNAME']
            streetKey = self.getStreetKeyFromFeatures(street)
            town = thing['properties']['ORTSBEZEICHNUNG']
            number = thing['properties']['HAUS_NR']
            if thing['properties']['HAUS_NR_ZUSATZ'] != ' ':
                number += thing['properties']['HAUS_NR_ZUSATZ']
            okh = thing['properties']['OKH']
            okv = thing['properties']['OKV']
            lat,lon = utm.to_latlon(okh, okv, 32, 'N')
            if streetKey in self.streets:
                self.streets[streetKey]['numbers'][number] = { 'lat':lat,
                                                    'lng':lon,
                                                    'town':town,
                                                   }
            else:
                self.streets[streetKey] = {'streetName':street,
                                        'numbers':{
                                           number:{'lat':lat,
                                                   'lng':lon,
                                                   'town':town,
                                                   } } }
        print(f"Num streets {len(self.streets)}")
        f = open('out.txt','w',encoding='utf-8')
        f.write('----------------- streets ------------------\n')
        for street in self.streets.keys():
            f.write(street+'\n')
        f.close()

    def getStreetKeyFromFeatures(self,street):
        street = street.lower()
        street = street.replace('-','')
        street = street.replace(' ','')
        if 'straße' in street:
            street = street.replace('straße','')
        return(street)

    def bestEffortNumber(self,streetKey):
        numChoices = list(self.streets[streetKey]['numbers'].keys())
        if len(numChoices) == 0:
            return None,None,None
        newNumber = random.choice(numChoices)
        lat = self.streets[streetKey]['numbers'][newNumber]['lat']
        lon = self.streets[streetKey]['numbers'][newNumber]['lng']
        town = self.streets[streetKey]['numbers'][newNumber]['town']
        return lat,lon,town

    def getTatortStuff(self,tatort):
        # tatort has either zero or one comma
        if tatort is None:
            return None,None,None,None,None
        neighborhood = None
        if ',' in tatort:
            neighborhood,rest = tatort.split(',')
        else:
            rest = tatort
        things = rest.split(' ')
        number = self.getNumber(things)
        if number:
            self.numNumber += 1
        streetKey = self.getStreetKeyFromData(things)
        lat = lon = town = None
        if not streetKey:
            streetNameNum = None
            self.numNoKey += 1
        else:
            streetNameNum = self.streets[streetKey]['streetName']
            if number: 
                streetNameNum += ' '+number
                self.numKeyAndNumber += 1
                if number in self.streets[streetKey]['numbers']:
                    lat = self.streets[streetKey]['numbers'][number]['lat']
                    lon = self.streets[streetKey]['numbers'][number]['lng']
                    town = self.streets[streetKey]['numbers'][number]['town']
                    self.numFullMatch += 1
                else:
                    lat,lon,town = self.bestEffortNumber(streetKey)
                    if lat:
                        self.bestEffort += 1
                    #print(number)
                    pass
            self.numFoundKey += 1
        # street+number, town, neighborhood, lat, lon
        return lat, lon, town, streetNameNum, neighborhood

    def getStreetKeyFromData(self,thing,p=False):
        if p: print(thing)
        for tryFrom in [0,1]:
            streetKey = ''
            for term in thing[tryFrom:]:
                if p: print(f"term = {term}")
                term = term.lower()
                if term == 'ecke':
                    term = ''
                term = term.replace('-','')
                term = term.replace('straße','')
                term = term.replace('strasse','')
                term = term.replace('staße','')
                if 'str.' in term:
                    index = term.find('str.')
                    term = term[:index]
                if term[-3:] == 'str':
                    term = term[:-3]
                term = term.replace('nr.','')
                term = term.replace('.ggü.','')
                streetKey += term
                if p: print(f"Try {streetKey}")
                if streetKey in self.streets:
                    return streetKey
        return None

    def getNumber(self,things):
        numbers = []
        number = None
        for thing in things:
            res = any(chr.isdigit() for chr in thing)
            if res:
                # clean out initial non-digit characters
                for i in range(len(thing)):
                    c = thing[i]
                    if c.isdigit():
                        break
                if i != 0:
                    thing = thing[i:]
                if '/' in thing:
                    splitNums = thing.split('/')
                    numbers.append(splitNums[-1])
                else:
                    numbers.append(thing)
        if len(numbers) > 1 and len(numbers[0]) == 2 and numbers[0][0] == 'c':
            del numbers[0]
        if len(numbers) == 1:
            number = numbers[0]
        elif len(numbers) > 1:
            number = numbers[-1]
        if number:
            number = number.replace('-','')
        return number

class getFeatures():
    def __init__(self):
        pass

pp = pprint.PrettyPrinter(indent=4)

fineTypes = ['fliessender','ruhender']
fineNum = {'fliessender':0,'ruhender':0}

homeDir = os.path.join(os.sep,'paul','tables','moers')
#print(homeDir)

featuresPath = os.path.join(homeDir,'Adressen_Kreis_Wesel_JSON.json')
feat = features(featuresPath)

rows = []
basePath = os.path.join(homeDir,'base_bussgelder.csv')
f = open(basePath,'r',encoding='utf-8')
reader = csv.DictReader(f)
for row in reader:
    rows.append(row)
f.close()
print(f"{len(rows)} rows in the data")

# Let's work on street names
# The following is some stats to figure out the best
# to use between Tatort and Tatort2. The results are
# below this
if False:
    res = {}
    for row in rows:
        t1str = ''
        t2str = '; '
        t1 = row['Tatort']
        if t1 is None: t1 = ''
        t1 = t1.lower()
        t2 = row['Tatort2']
        if t2 is None: t2 = ''
        t2 = t2.lower()
        if len(t2) < 3 and len(t1) < 3:
            t1str += 'short, '
            t2str += 'short, '
            continue
        elif len(t2) < 3:
            t2str += 'short, '
        elif len(t1) < 3:
            t1str += 'short, '
        t2Commas = t2.count(',')
        if t2Commas > 1:
            # There are very few of these, so ignore
            t2str += '>1 commas, '
            # continue
        if t2Commas == 0:
            t2str += '0 commas, '
            #if t2[:2] not in ['fr','im','am','an','in'] and t2[2] == ' ':
                # Also few of these, ignore
                #continue
        if t2Commas == 1:
            t2str += '1 comma, '
        t1Commas = t1.count(',')
        if t1Commas > 1:
            # There are very few of these, so ignore
            t1str += '>1 commas, '
            # continue
        if t1Commas == 0:
            t1str += '0 commas, '
        if t1Commas == 1:
            t1str += '1 comma, '
        allStr = t1str + t2str
        if len(t1) > len(t2):
            allStr += '; t1 longer'
        else:
            allStr += '; t2 longer'
        if allStr in res:
            res[allStr]['num'] += 1
            res[allStr]['t1Avg'] + len(t1)
            res[allStr]['t1cnt'] + 1
            res[allStr]['t2Avg'] + len(t2)
            res[allStr]['t2cnt'] + 1
        else:
            res[allStr] = {'num':1, 't1Avg':len(t1), 't1cnt':1,
                                    't2Avg':len(t2), 't2cnt':1 }
    for k,v in res.items():
        v['t1Avg'] = round(v['t1Avg'] / v['t1cnt'])
        v['t2Avg'] = round(v['t2Avg'] / v['t2cnt'])
    pp.pprint(res)
    quit()
'''
{   '0 commas, ; 0 commas, ; t1 longer': {   'num': 1,
                                             't1Avg': 5,      
                                             't1cnt': 1,      
                                             't2Avg': 3,      
                                             't2cnt': 1},     
    '0 commas, ; 0 commas, ; t2 longer': {   'num': 71718,    
                                             't1Avg': 5,      
                                             't1cnt': 1,      
                                             't2Avg': 32,     
                                             't2cnt': 1},     
    '0 commas, ; 1 comma, ; t2 longer': {   'num': 55622,     
                                            't1Avg': 5,       
                                            't1cnt': 1,       
                                            't2Avg': 24,      
                                            't2cnt': 1},      
    '0 commas, ; >1 commas, ; t2 longer': {   'num': 24,      
                                              't1Avg': 5,     
                                              't1cnt': 1,     
                                              't2Avg': 55,    
                                              't2cnt': 1},    
    '1 comma, ; 0 commas, ; t1 longer': {   'num': 223672,    
                                            't1Avg': 38,      
                                            't1cnt': 1,       
                                            't2Avg': 16,      
                                            't2cnt': 1},      
    '1 comma, ; 0 commas, ; t2 longer': {   'num': 88,        
                                            't1Avg': 34,      
                                            't1cnt': 1,       
                                            't2Avg': 36,      
                                            't2cnt': 1},      
    '1 comma, ; short, 0 commas, ; t1 longer': {   'num': 107,
                                                   't1Avg': 28,
                                                   't1cnt': 1,
                                                   't2Avg': 2,
                                                   't2cnt': 1},
    '>1 commas, ; 0 commas, ; t1 longer': {   'num': 49,
                                              't1Avg': 35,
                                              't1cnt': 1,
                                              't2Avg': 14,
                                              't2cnt': 1},
    'short, 0 commas, ; 0 commas, ; t2 longer': {   'num': 185409,
                                                    't1Avg': 0,
                                                    't1cnt': 1,
                                                    't2Avg': 26,
                                                    't2cnt': 1},
    'short, 0 commas, ; 1 comma, ; t2 longer': {   'num': 44734,
                                                   't1Avg': 0,
                                                   't1cnt': 1,
                                                   't2Avg': 47,
                                                   't2cnt': 1},
    'short, 0 commas, ; >1 commas, ; t2 longer': {   'num': 3,
                                                     't1Avg': 0,
                                                     't1cnt': 1,
                                                     't2Avg': 50,
                                                     't2cnt': 1}}
'''
for row in rows:
    t1 = row['Tatort']
    if t1 is None: t1 = ''
    t1 = t1.lower()
    t2 = row['Tatort2']
    if t2 is None: t2 = ''
    t2 = t2.lower()
    t1Commas = t1.count(',')
    t2Commas = t2.count(',')
    t1Short = False
    if len(t1) < 3: t1Short = True
    t2Short = False
    if len(t2) < 3: t2Short = True
    if ((t1Short and t2Commas <= 1) or
        (t1Commas == 0 and t2Commas == 1) or
        (t1Commas == 0 and t2Commas == 0 and len(t2) > len(t1))):
        lat, lon, town, streetnum, neighborhood = feat.getTatortStuff(t2)
    elif (t1Commas == 1 and t2Commas == 0):
        lat, lon, town, streetnum, neighborhood = feat.getTatortStuff(t1)
    else:
        lat, lon, town, streetnum, neighborhood = feat.getTatortStuff(None)
    row['StrasseNum'] = streetnum
    row['lat'] = lat
    row['lng'] = lon
    row['Ortsbezeichnung'] = town
    row['Nachbar'] = neighborhood
print(f"Found {feat.numFoundKey} keys, missed {feat.numNoKey} keys")
print(f"Numbers: {feat.numNumber}, key and number: {feat.numKeyAndNumber}, full match: {feat.numFullMatch}")
print(f"Best Effort {feat.bestEffort}")

fieldnames = ['PID',
        'Nummernschild',
        'BussgelderTyp',
        'Geldbusse',
        'TatbNr',
        'Tatort',
        'Tatort2',
        'StrasseNum',
        'Ortsbezeichnung',
        'Nachbar',
        'Tattag',
        'Zeit',
        'Tbnr1',
        'lat',
        'lng',
]
csvPath = os.path.join(homeDir,'finalMoers.csv')
with open(csvPath, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)