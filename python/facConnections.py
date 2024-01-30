import requests
import pprint
import itertools

'''
This builds a graph of co-author relations between faculty at MPI-SWS
using data from DBLP.

It produces a file called facGraph.txt which contains a graph usable
by Graphviz. To draw the graph, go to
http://magjac.com/graphviz-visual-editor/ and copy the contents of
facGraph.txt into the box.

The 'pid' field is the DBLP identifier for the author.
'''

pp = pprint.PrettyPrinter(indent=4)

# https://dblp.org/pid/19/2942
baseUrl = 'https://dblp.org/pid/'

facs = {
    'Keon Jang':{'pid':'40/503',
    'status':'left', 'ini':'KJ'},
    'Eva Darulova':{'pid':'75/10322',
    'status':'left', 'ini':'ED'},
    'Björn Brandenburg':{'pid':'19/2942',
    'status':'active', 'ini':'BB'},
    'Maria Christakis':{'pid':'05/7730',
    'status':'left', 'ini':'MC'},
    'Derek Dreyer':{'pid':'d/DerekDreyer',
    'status':'active', 'ini':'DD'},
    'Peter Druschel':{'pid':'d/PDruschel',
    'status':'active', 'ini':'PD'},
    'Paul Francis':{'pid':'f/PaulFrancis',
    'status':'active', 'ini':'PF'},
    'Deepak Garg':{'pid':'45/6786-1',
    'status':'active', 'ini':'DG'},
    'Manuel Gomez Rodriguez':{'pid':'73/8260',
    'status':'active', 'ini':'MR'},
    'Krishna Gummadi':{'pid':'g/PKrishnaGummadi',
    'status':'active', 'ini':'KG'},
    'Antoine Kaufmann':{'pid':'145/5081',
    'status':'active', 'ini':'AK'},
    'Jonathan Mace':{'pid':'154/0937',
    'status':'left', 'ini':'JM'},
    'Rupak Majumdar':{'pid':'71/1981',
    'status':'active', 'ini':'RM'},
    'Joel Ouaknine':{'pid':'55/4663',
    'status':'active', 'ini':'JO'},
    'Adish Singla':{'pid':'58/657',
    'status':'active', 'ini':'AS'},
    'Viktor Vafeiadis':{'pid':'69/1549',
    'status':'active', 'ini':'VV'},
    'Georg Zetzsche':{'pid':'24/651',
    'status':'active', 'ini':'GZ'},
}

print("Getting all the author rdf docs from DBLP")
for name,stuff in facs.items():
    print(f"    Getting doc for {name}")
    url = baseUrl + stuff['pid'] + '.rdf'
    r = requests.get(url)
    facs[name]['rdf'] = r.text
    print(f"        Got {len(facs[name]['rdf'])} bytes")

print("\nExtracting publications")
pubs = {}
for name,stuff in facs.items():
    print(f"    Extracting pubs for {name}")
    lines = stuff['rdf'].splitlines()
    print(f"{len(lines)} lines")
    for line in lines:
        print(line)
        if 'authorOf' in line:
            paper = line.split('"')[1]
            if paper in pubs:
                pubs[paper].append(stuff['ini'])
            else:
                pubs[paper] = [stuff['ini']]

print(f"    Total {len(pubs)} publications found")

print("\nCompute co-author graph")
pairs = {}
for pub,authors in pubs.items():
    if len(authors) == 1:
        continue
    for comb in itertools.combinations(authors,2):
        sortedAuthors = sorted(list(comb))
        pairKey = f'{sortedAuthors[0]} -- {sortedAuthors[1]}'
        if pairKey in pairs:
            pairs[pairKey] += 1
        else:
            pairs[pairKey] = 1
pp.pprint(pairs)

print("\nGenerate graph suitable for Graphviz")
graphviz = '''
graph ER {
    splines=false
    pack=false
	fontname="Helvetica,Arial,sans-serif"
	node [fontname="Helvetica,Arial,sans-serif"]
	edge [fontname="Helvetica,Arial,sans-serif"]
	layout=neato
    node [shape=circle,fontsize=10]; '''
for name,stuff in facs.items():
    if stuff['status'] == 'active':
       graphviz += stuff['ini'] + '; '

graphviz += '\n    node [shape=circle,fontsize=10,style=filled,color=lightgray]; '''
for name,stuff in facs.items():
    if stuff['status'] == 'left':
       graphviz += stuff['ini'] + '; '
graphviz += '\n'
for link,cnt in pairs.items():
    graphviz += f'    {link} [penwidth={cnt},label="{cnt}",fontsize=10,color=cadetblue2];\n'

graphviz += '\nlabel = "Faculty co-author, Sept. 2022"\n}'

f = open('facGraph.txt','w')
f.write(graphviz)
f.close()
