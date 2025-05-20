def binaryIteration(n:int):
    yield (0,1)
    for x  in range(1,n):
        i=0
        axis=x
        while axis%2==0:
            i+=1
            axis>>=1
            xm1=x-(1<<i)
            yield (xm1,x)
        yield (x,x+1)
print(list(binaryIteration(10)))

