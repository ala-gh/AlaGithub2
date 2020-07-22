a = 3456
len_a = len(str(a))
#a =str(a)
#a = a.split()
print(a)
bagh = []
for i in range(len_a):
    kh = a//10
    ba = a%10
    bagh.append(ba)
    a = kh