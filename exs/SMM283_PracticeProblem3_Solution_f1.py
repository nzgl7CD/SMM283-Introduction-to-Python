##Draft solution by Dr. Adrian Euler
#-------------------------------------------------

def power(r, n): ##r^n = r*r^(n-1)
    temp = 1
    if n < 0:
        r = 1.0/r
        n = -n
    if n == 0:
        return temp
    if n > 0:
        for i in range(1, n+1):
            temp = temp*r
    return temp
#-------------------------------------------
def powerv2(r, n):  #alternative solution
    if n < 0:
        r = 1.0/r
        n = -n
    if n == 0:
        return(1)
    return (r *powerv2(r, n-1)) ##r^n = r*r^(n-1)

# This function uses recursion
def factorial(n):
    if n == 0:
        return (1)
    if n > 0:
        return (n*factorial(n-1))   ##n! = n*(n-1)!
#----------------------------------------------------
def factorial_(n):  #alternative solution
    temp = 1
    if n == 0:
        return (1)
    if n > 0:
        for i in range(n, 0, -1):
            temp = temp*i   ##n! = n*(n-1)!
    return(temp)
#----------------------------------------------------
def e(x):
    r = 0.0
    tol = 0.01
    n = int(1.0/tol)
    for i in range(n):
        r += (math.pow(x, i)/factorial(i))
    return r
#---------------------------------------------------

def u(sigma, T, nsteps):
    dt = T/float(nsteps)
    x = sigma*(dt**0.5)
    r = e(x)
    return r
#---------------------------------------------------
def p(r, q, sigma, T, nsteps):
    up = u(sigma, T, nsteps)
    down = 1/up
    dt = T/nsteps
    x = (r-q)*dt
    prob = (e(x)-down)/(up-down)
    return prob
#-------------------------------------------------
def combin(n, k):
    nf = factorial(n)
    kf = factorial(k)
    n_k = factorial(n-k)
    c = nf/(kf*n_k)
    return c
#------------------------------------------------
def pdf(r, q, sigma, T, nsteps):
    prob = p(r, q, sigma, T, nsteps)
    pstar = 1 - prob
    k = []
    for i in range(nsteps, -1, -1):
        k.append(i)
    r = []
    temp = 0.0
    for  j in range(nsteps, -1, -1):
        temp = combin(nsteps, k[j])*(power(prob, k[j])*power(pstar, (nsteps-k[j])))
        r.append(temp)
    return r
#-------------------------------------------------------------------------------------
def optval(r, T, payoff = [], prob = []):
    tot = 0.0
    n = len(prob)
    for i in range(n):
        tot += (payoff[i]*prob[i])
        #print(payoff[i], "\t", prob[i])
    return tot*e(-r*T)
#------------------------------------------------------------------------------------
#The part below is provided by Dr. Adrian Euler
#stdents should use to test their user-defined functions.
#students should not modify the main.
import math
from datetime import datetime
import time as clock

def main():
    start = clock.time()
    message = \
    '''
      \t-----------------------------------------------------------------------------------
      \tIndivual test for SMM283 - Introduction to Python
      \t----------------------------------------------------------
      \tStudent should test the user-defined functions using \n\tthe main function provided here.
      \tMake sure you enter your student name and student id. \n\tDo not enter the name/student id
      \tof anyone else...
      \tA program password will be provided at the start of the test. When prompted
      \tby the program enter the correct password.
      \tDo NOT change the code of the main provided.
      \tIf your code runs with the main provided, then it passes\n\t the basic tests.
      \tCode will be inspected further...
      \t-----------------------------------------------------------------------------------
    '''
    print(message)
    s = 100.0      #share price
    x = 95.0       #exercise price
    r = 0.08       #short rate
    q = 0.03       #dividend yield
    T = 0.5        #option expiration
    sigma = 0.20   #volatility
    nsteps = 9          #number of steps in the bionomial tree
    iopt = 1       #i-option (+1 for call, -1 for put)
    while True:
        name = input("\tEnter your full name: ")
        yn = input("\t\tDid you enter your name correctly (y or n): ")
        if yn.lower()=='y':
            break
        else:
            continue
    while True:
        stdId = input("\tEnter your (student) ID: ")
        yn = input("\t\tDid you enter your student ID correctly (y or n): ")
        if yn.lower()=='y':
            break
        else:
            continue
    while True:
        pwd = input("\tEnter program pass (provided at the start of the test): ")
        if pwd == "*********":
            print("\t\tProgram pass is OK!")
            break
        else:
            print("\t\tProgram pass did NOT match! ")
            continue
    payvals = [57.8465, 44.0945, 31.5797, 20.1910, 9.8269, 0.3953, 0.0000, 0.0000, 0.0000, 0.0000]
    pvals = [0.0027, 0.0224, 0.0835, 0.1814, 0.2535, 0.2362, 0.1467, 0.0586, 0.0136, 0.0014]

    result1 = factorial(nsteps)
    print("\n\t------------------------------------------------------------")
    print("\t\t\t", name, "/", stdId, "({0})".format(pwd), ":")
    print("\t-------------------------------------------------------------")
    if result1 == math.factorial(nsteps):
        print("\t1.Your user-defined function \"factorial(...)\" passed the test!")
    else:
        print("\t1.Incorrect result.Your user-defined function \"factorial(...)\" did not passed the test!")
    result2 = round(e(4.5), 4)
    if result2 == round(math.exp(4.5), 4):
         print("\t2.Your user-defined function \"e(...)\" passed the test!")
    else:
        print("\t2. Incorrect result.Your user-defined function \"e(...)\" did not passed the test!")

    result3 = round(u(sigma, T, nsteps), 4)
    if result3 == 1.0483:
         print("\t3.Your user-defined function \"u(...)\" passed the test!")
    else:
        print("\t3. Incorrect result.Your user-defined function \"u(...)\" did not passed the test!")
   
    result4 = round(p(r, q, sigma, T, nsteps), 4)
    if result4 == 0.5177:
         print("\t4.Your user-defined function \"p(...)\" passed the test!")
    else:
        print("\t4. Incorrect result.Your user-defined function \"p(...)\" did not passed the test!")
   
    result5 = combin(10, 4)
    if result5 == 210.00:
         print("\t5.Your user-defined function \"combin(...)\" passed the test!")
    else:
        print("\t5. Incorrect result.Your user-defined function \"combin(...)\" did not passed the test!")
    i = 9
    result6 = pdf(r, q, sigma, T, nsteps)
    for i in range(nsteps, 0, -1):
        if round(result6[i], 4) == round(pvals[nsteps-i], 4):
            print((nsteps-i+1)*"\t", "({0})---{1}".format(nsteps-i,"Match!"))
            i -= 1    
        else:
            print((nsteps-i+1)*"\t", "({0})---{1}".format(nsteps-i,"NO Match!"))
            break
            
    if i == 0:
        print("\t6.Your user-defined function \"pdf(...)\" passed the test!")
    else:
        print("\t6. Incorrect.Your user-defined function \"pdf(...)\" did not passed the test!")
    result7 = round(optval(r, T, payvals, pvals), 2)
    if result7 == 9.63:
         print("\t7.Your user-defined function \"optval(...)\" passed the test!")
         print("\n\t---------------------------------------------------------------------")
         print("\tFor the data given, the binomial call option price is: £{0} ".format(result7))
         print("\t-----------------------------------------------------------------------")
    else:
        print("\t7. Incorrect result.Your user-defined function \"optval(...)\" did not passed the test!")
    now = datetime.now()
    dt = now.strftime("%d/%m/%Y %H:%M:%S")
    print("\n\tTested by {0} with ID {1} on date and time {2} ".format(name, stdId, dt))
    end = clock.time()
    print("\n\tIt took {0}seconds to run the program!".format(end-start))
    input("\n\n\tPress ENTER Key to Exit!")
main()
