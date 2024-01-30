import rpyc
import time

if __name__ == "__main__":
    c = rpyc.classic.connect("localhost")
    c1 = rpyc.classic.connect("localhost",port=20000)
    print(c.modules)
    print(c1.modules)
    print(c.namespace)
    rsys = c.modules.sys
    rsys1 = c1.modules.sys
    ros = c.modules.os
    ros1 = c1.modules.os
    print(ros.getcwd())
    path = ros.path.join('C:\\','paul','GitHub','attacker','diffixElmPaperAttacks','tools')
    path1 = ros1.path.join('C:\\','paul','GitHub','attacker','diffixElmPaperAttacks','tools')
    rsys.path.append(path)
    rsys1.path.append(path)
    ros.chdir(path)
    print(ros.getcwd())
    sc = c.modules.score.score
    score = sc()
    print(ros1.getcwd())
    score.attempt(score,1,1,1)
    print(score.attempts)
    score.attempt(score,1,1,1)
    print(score.attempts)
    print(score.computeScore())
    print('-------------')
    a_attempt = rpyc.async_(score.attempt)
    res = a_attempt(score,1,1,1)
    res = a_attempt(score,1,1,1)
    res = a_attempt(score,1,1,1)
    print(score.attempts)
    print(res)
    print(f"ready: {res.ready}")
    print(f"error: {res.error}")
    print(f"value: {res.value}")
    a_compute = rpyc.async_(score.computeScore)
    res = a_compute()
    print(f"ready: {res.ready}")
    print(f"error: {res.error}")
    print(f"value: {res.value}")
    res = a_attempt(score,1,1,0)
    res = a_compute()
    print(f"ready: {res.ready}")
    print(f"error: {res.error}")
    print(f"value: {res.value}")