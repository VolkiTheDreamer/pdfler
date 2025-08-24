import pandas as pd

df=pd.read_csv("all_ratings.txt",sep="\t")
df=df.groupby(["title","genres"])["rating"].mean().to_frame().reset_index("genres")

uniqs=df.genres.unique()
uniqs1=[x.split("|") for x in uniqs]
uniqs2 = set([x.lower() for sub in uniqs1 for x in sub])

for u in uniqs2:
    df[u]=df.genres.apply(lambda x: u in x.lower())

genre= input("Enter a genre: ").lower()
maxrating=df[df[genre]]["rating"].sort_values(ascending=False)

m=0
while m<len(maxrating):
    cevap=input(f"Would you like to watch {maxrating.index[m]} with rating {maxrating[m]}: ").lower()
    if cevap=="yes":
        print("Enjoy the movie!")
        break
    elif cevap=="no":
        m=m+1
        # pass
    else:
        print("You entered an incorrect input, please enter yes or no: ")
        
else:
    print("No movies left in this genre!")

