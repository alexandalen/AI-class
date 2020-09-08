# 方程式語法 (tkinter)

from sympy.solvers import solve
import tkinter as tk

windows = tk.Tk()
windows.title("聯立方程式計算")
windows.geometry("400x300")
windows.configure(background="white")

def run():
    equation=[]
    expr1=text1.get("1.0","end")
    equat=expr1.split("\n")
    if len(equat)>0:
        for j in range(len(equat)):
            if equat[j]=="":
                continue
            print (equat[j])
            arr=equat[j].split("=")
            if len(arr)==2:
                equat[j]=arr[0] + "-(" + arr[1] + ")"
                equation.append(equat[j])
        print(equation)
        answer1.set(solve(equation))

answer1 = tk.StringVar()
header_label=tk.Label(windows, text="解聯立方程式",height=1, font=("標楷體",20,"bold"))
header_label.pack()
text1=tk.Text(windows, height=4, width=30, font=("Comic Sans MS", 18,"bold"))
text1.pack()
Bt=tk.Button(windows, text="解答", font=("標楷體", 36,"bold"), command=run)
Bt.pack()
Lb=tk.Label(windows, text="",height=1, textvariable=answer1, font=("Comic Sans MS", 18,"bold") )
Lb.pack()

windows.mainloop()

