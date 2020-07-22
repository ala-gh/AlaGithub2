#***************************1**************************************
# : masahate motevazi o alazlaa!!!
# a = input("length :")
# a = float(a)
#
# b = input("high :")
# b = float(b)
#
# c = a*b
# print(a,",",b,"area:","=",c)

#*************************2************************************
import numpy as np
#
# aa =[]
# a = int(input('\n enter n average :'))
# for i in range(a):
#     aaa=int(input())
#     aa.append(aaa)
# print(aa)
# amax = max(aa)
# print(amax)
# aa.remove(amax)
# amax2 = max(aa)
# print(amax2)

#*****************************3***************************
# adad = int(input("enter adad :"))
# ragham = int(input("enter ragham :"))
# len_adad = len(str(adad))
# baghi = []
# adad2=adad
#
# for i in range(len_adad):
#     kharej = adad2//10
#     ba = adad2% 10
#     baghi.append(ba)
#     adad2 = kharej
# t=0
# for i in range(len(baghi)):
#     if ragham == baghi[i]:
#         t=t+1
# print(t)
#*********************************** 4 ************************
# def add_list1(x):
#     sum =0
#     for i in range(len(x)):
#         sum =sum +x[i]
#
#     return sum
# a = [1,5,6,7,8]
# print(a)
# summ=add_list1(a)
# print(summ)

#*********************************** 5 - 9 min ***************************
list1 = [2,4,8,4,6]
list2 =['a','u','t']
list3 =[]
len_list2 = len(list2)

for i in range(len(list1)):
    list3.append(list1[i])
    if len_list2 !=0:
        list3.append(list2[i])
        len_list2 = len_list2-1

print(list3)



