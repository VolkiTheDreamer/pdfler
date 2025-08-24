from random import randint

temp1=0
temp2=0
for i in range(32):
    dice=randint(1,6)
    temp1+=dice

print("score 1: "+str(temp1))

if temp1>121:
    print("Player 2 wins!")
else:
    for i in range(32):
        dice=randint(1,6)
        temp2+=dice
    print("score 2: "+str(temp2))    

    if temp1>temp2:
        print("Player 1 wins!")
    elif temp1<temp2:
        if temp2<=121:
            print("Player 2 wins!")
        else:
            print("Player 1 wins!")
    else:
        print("There is a tie!")