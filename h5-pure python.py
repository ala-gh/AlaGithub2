#1: masahate motevazi o alazlaa!!!
# a = input("length :")
# a = float(a)
#
# b = input("high :")
# b = float(b)
#
# c = a*b
# print(a,",",b,"area:","=",c)

#2
import numpy as np

aa =[]
a = int(input('\n enter n average :'))
for i in range(a):
    aaa=int(input())
    aa.append(aaa)
print(aa)
amax = max(aa)
print(amax)
aa.remove(amax)
amax2 = max(aa)
print(amax2)



# # number of elements
# n = int(input("Enter number of elements : "))
#
# # Below line read inputs from user using map() function
# a = list(map(int, input("\nEnter the numbers : ").strip().split()))[:n]
#
# print("\nList is - ", a)

