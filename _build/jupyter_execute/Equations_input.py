# 方程式語法 (Input)

equation=[]
from sympy.solvers import solve
from sympy import symbols
while 1:
    a=input("請輸入方程式:")
    if a=="":
        break
    arr=a.split("=")
    if len(arr)==2:
        a=arr[0]+"-("+arr[1]+")"
        equation.append(a)
print(equation)
solve(equation) 

