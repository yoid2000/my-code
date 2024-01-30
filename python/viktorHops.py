import os
import pprint

pp = pprint.PrettyPrinter(indent=4)

def getAuthorPrestige():
    pass

def getCoauthors(coauth, aid, aidExclude=None):
    # coauth is a dict
    for aid2,cnt in coauthors[aid].items():
        isDirectCoauthor = False
        if aidExclude and aidExclude in coauthors[aid2]:
            isDirectCoauthor = True
        if aid2 in coauth:
            coauth[aid2]['cnt'] += cnt
        else:
            coauth[aid2] = {'name':authors[aid2], 'aid':aid2, 'cnt':1, 'isDirectCoauthor':isDirectCoauthor}
            pass

authPath = os.path.join(os.environ['AB_RESULTS_DIR'], 'ent.author')
graphPath = os.path.join(os.environ['AB_RESULTS_DIR'], 'out.dblp_coauthor')

viktor = '419683'
# This is Rodrigo and Allen
oneHopExclude = ['170071', '416788']
oneHopExclude = []

authors = {}
i = 0
with open(authPath, 'r', encoding='utf-8') as f:
    for line in f:
        #print(i)
        i += 1
        stuff = line.split()
        if len(stuff) == 2 and stuff[0].isdigit:
            # stuff[0] is aid and stuff[1] is name
            name = stuff[1].replace('_',' ')
            authors[stuff[0]] = name

print(f"authors table has {len(authors)} items")

coauthors = {}
with open(graphPath,'r') as f:
    for line in f:
        stuff = line.split()
        if len(stuff) == 4:
            aid1 = stuff[0]
            aid2 = stuff[1]
            if aid1 in oneHopExclude or aid2 in oneHopExclude:
                continue
            dic = stuff[3]
            if aid1 not in coauthors:
                coauthors[aid1] = {aid2:1}
            else:
                if aid2 not in coauthors[aid1]:
                    coauthors[aid1][aid2] = 1
                else:
                    coauthors[aid1][aid2] += 1
print(f"coauthors table has {len(coauthors)} items")
print("Viktor's coauthors:")
coauth = {}
getCoauthors(coauth, viktor)
pp.pprint(coauth)

cocoauth = {}
for aid2,cnt in coauthors[viktor].items():
    getCoauthors(cocoauth, aid2, aidExclude=viktor)
print("Coauthors of Viktor's coauthors:")
pp.pprint(cocoauth)

# We are going to gauge "prestige" by the number of coauthor

print(f"Viktor has {len(coauth)} coauthors and {len(cocoauth)} co-coauthors")
nameCount = [[x['name'],x['cnt'],x['isDirectCoauthor']] for x in cocoauth.values()]
nameCount = sorted(nameCount, key=lambda x:x[1], reverse=True)
pp.pprint(nameCount)
