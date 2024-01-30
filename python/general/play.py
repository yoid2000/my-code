from typing import List, Dict, Optional

def thing(dump: Optional[None | List[Dict]] = None, count: Optional[int] = 0) -> List[Dict]:
    dump = [] if dump is None else dump
    dump.append({'a':'b'})
    count += 1
    if count > 10:
        return dump
    return thing(dump = dump, count = count)

dump = thing()
print(dump)