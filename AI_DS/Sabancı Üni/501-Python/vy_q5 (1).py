#iris dosyasının sonundaki boş satır list comprehension sırasında problem yaratıyordu, manuel sildim

with open("iris.txt","r") as dosya:
    content=dosya.read()

lines=[x for x in content.split("\n")]

setosa_data=[(x.split(",")[0],x.split(",")[1]) for x in lines if x.split(",")[4]=="setosa"]
virginica_data=[(x.split(",")[0],x.split(",")[1]) for x in lines if x.split(",")[4]=="virginica"]
versicolor_data=[(x.split(",")[0],x.split(",")[1]) for x in lines if x.split(",")[4]=="versicolor"]

plt.scatter([x[0] for x in setosa_data],[x[1] for x in setosa_data])
plt.scatter([x[0] for x in setosa_data],[x[1] for x in virginica_data])
plt.scatter([x[0] for x in setosa_data],[x[1] for x in versicolor_data])


#tam olmadı ama hatanını nerde odlugunu anlayamadım