maclar=skorlar.split("\n")
puanlar={}
for m in maclar:    
    ts=m.split("-")
    left=ts[0]; t1=left[:-2]; t1_s=left[-1]
    right=ts[1]; t2=right[2:]; t2_s=right[0]
    if t1_s>t2_s:        
        if t1 in puanlar:
            puanlar[t1]=puanlar[t1]+3
        else:
            puanlar[t1]=3
    elif t1_s<t2_s:
        if t2 in puanlar:
            puanlar[t2]=puanlar[t2]+3
        else:
            puanlar[t2]=3
    else:
        if t1 in puanlar:
            puanlar[t1]=puanlar[t1]+1
        else:
            puanlar[t1]=1
        if t2 in puanlar:
            puanlar[t2]=puanlar[t2]+1
        else:
            puanlar[t2]=1

for k,v in puanlar.items():
    print(k,v)