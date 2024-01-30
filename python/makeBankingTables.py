import psycopg2
import random
import pprint
import pandas as pd
import sqlalchemy as sq

'''
This is a one-off script used to create a banking database suitable
for Open Diffix (and multiple AID columns but no JOIN in particular)

It creates a table that brings together account and client data, and
can subsequently be used to create the other tables (with JOIN).

Note that if this is run on Windows, you may have to do:
    iconv -f UTF-16 -t UTF-8 commands > commands8

On the linux machine so that `\i commands8` works in psql
(where `commands` is the file produced by this script)
'''


def getZip():
    zips = [12341,12340,12345,12346,12347,12278,12279,12289,12281,23456,23457,23458,23459,23898]
    ran = int(random.gauss(7,1.5))
    if ran >= len(zips): ran = len(zips)-1
    return zips[ran]

pp = pprint.PrettyPrinter(indent=4)

def testGetZip():
    test = {}
    for _ in range(1000):
        zip = getZip()
        if zip in test:
            test[zip] += 1
        else:
            test[zip] = 1
    pp.pprint(test)
    print(len(test))
    
if False:
    testGetZip()
    quit()

connStr = str(
            f"host={'db001.gda-score.org'} port={5432} dbname={'banking'} user={'direct_user'} password={'demo'}")
conn = psycopg2.connect(connStr)
cur = conn.cursor()

sql = '''
select * from accounts where account_id = 2749;
'''
cur.execute(sql)
ans = cur.fetchall()
#pp.pprint(ans)

sql = 'select * from accounts'
cur.execute(sql)
allAcct = cur.fetchall()
#sql = 'select * from clients'
#cur.execute(sql)
#allCli = cur.fetchall()

acctCols = ['account_id', 'acct_district_id', 'frequency', 'acct_date', 'client_id', 'disp_type', 'birth_number', 'cli_district_id', 'lastname']
#print(acctCols)
acctIacct = acctCols.index('account_id')
acctIcli = acctCols.index('client_id')

cliCols = ['client_id', 'birth_number', 'district_id', 'lastname']
#print(cliCols)
cliIcli = cliCols.index('client_id')

accounts = {}
for row in allAcct:
    aid = row[acctIacct]
    clientInfo = {
        'client_id':row[4],
        'disp_type':row[5],
        'birth_number':str(row[6]),
        'cli_district_id':row[7],
        'lastname':row[8],
    }
    if aid in accounts:
        accounts[aid]['clients'].append(clientInfo)
    else:
        accounts[aid] = {
            'account_id':row[0],
            'acct_district_id':row[1],
            'frequency':row[2],
            'acct_date':str(row[3]),
            'clients':[clientInfo],
        }

if False:
    accounts = {}
    for row in allAcct:
        print(row)
        aid = row[acctIacct]
        accounts[aid] = row
        quit()

    clients = {}
    for row in allCli:
        cid = row[cliIcli]
        clients[cid] = row

sql = '''
DROP TABLE IF EXISTS joined_clients;
'''
print(sql)

sql = '''
CREATE TABLE joined_clients (
    account_id        integer,
    acct_district_id  integer,
    frequency         text,
    acct_date         text,
    client_id1        integer,
    cli_district_id1  integer,
    disp_type1        text,
    birth_number1     text,
    lastname1         text,
    client_id2        integer,
    cli_district_id2  integer,
    disp_type2        text,
    birth_number2     text,
    lastname2         text
);
'''
print(sql)

for aid,s in accounts.items():
    #print(aid,cids)
    c = s['clients']
    numClients = len(c)
    if numClients == 1 and c[0]['disp_type'] != 'OWNER':
        print("bad owner")
        pp.pprint(s)
        quit()
    if numClients == 2 and c[0]['disp_type'] == c[1]['disp_type']:
        print("bad disp types")
        pp.pprint(s)
        quit()
    acctStr = f'''
        {s['account_id']},
        {s['acct_district_id']},
        '{s['frequency']}',
        '{s['acct_date']}',
    '''
    ownIndex = 0
    otherIndex = 1
    if numClients == 2:
        if c[1]['disp_type'] == 'OWNER':
            ownIndex = 1
            otherIndex = 0
        cli2Str = f'''
            {c[otherIndex]['client_id']},
            {c[otherIndex]['cli_district_id']},
            '{c[otherIndex]['disp_type']}',
            '{c[otherIndex]['birth_number']}',
            '{c[otherIndex]['lastname']}'
        '''
    else:
        cli2Str = f'''
            NULL,
            NULL,
            NULL,
            NULL,
            NULL
        '''
    cli1Str = f'''
        {c[ownIndex]['client_id']},
        {c[ownIndex]['cli_district_id']},
        '{c[ownIndex]['disp_type']}',
        '{c[ownIndex]['birth_number']}',
        '{c[ownIndex]['lastname']}',
    '''
    sql = f'''INSERT INTO joined_clients VALUES ({acctStr}{cli1Str}{cli2Str});'''
    print(sql)
    pass


#engine = sq.create_engine("postgresql+psycopg2://gda-score_ro_user:moquaiR7@db001.gda-score.org:5432/raw_banking")
#dfclients = pd.read_sql_table('clients',engine)
quit()

# Quit development at this point. Not sure updating the banking dataset is
# really worth doing...



sql = '''
    SELECT
        column_name
    FROM
        information_schema.columns
    WHERE
        table_schema = 'public' AND 
        table_name = 'accounts'
'''
cur.execute(sql)
ans = cur.fetchall()
pp.pprint(ans)