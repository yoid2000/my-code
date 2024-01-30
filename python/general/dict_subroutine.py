
def mod_dict(x):
    x[1][1] = 9

def mod_list(x):
    x[1] = 10

l = [1,2,3,4]
print(l)
mod_list(l)
print(l)

d = {1:[1,1,1], 2:[2,2,2]}
print(d)
mod_dict(d)
print(d)