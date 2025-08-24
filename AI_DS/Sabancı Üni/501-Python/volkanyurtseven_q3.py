import string
pwd=input("Enter a pass: ")
print(checkPass(pwd))

def checkPass(pwd):
    if len(pwd)>=8 and len(set(pwd).intersection(set(string.digits)))>0 and len(set(pwd).intersection(set(['_', '@', '&', '%', '*'])))>0:
        return True
    else:
        return False    