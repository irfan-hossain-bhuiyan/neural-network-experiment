def a(c):
    def cInc():
        nonlocal c
        c+=1
        print(c)
    return cInc
s=a(4)
s()
s()
d=a(20)
s()

