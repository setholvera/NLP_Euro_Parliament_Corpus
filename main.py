import sys
import os
import random

cleaning=10000 # 0 to deactivate clean
training=100000
test=10000
bigram=True
GENERATECORPUS=True

def normalize(line):
    r=""
    for s in line:
        if s.isalpha():
            r=r+s
        else:
            r=r+" "
    words=r.strip().split(' ')
    words=[w for w in words if words!=""]
    if not bigram:
        return words
    else:
        r=[]
        for i in range(len(words)-1):
            r.append(words[i]+"_"+words[i+1])
        return r



def save(id,map):
    import pickle
    path=id+'.dictionary'
    if bigram:
        path+="2"
    with open(path, 'wb') as config_dictionary_file:
        pickle.dump(map, config_dictionary_file)

def load(id):
    import pickle
    path=id+'.dictionary'
    if bigram:
        path+="2"
    with open(path, 'rb') as config_dictionary_file:
        return pickle.load(config_dictionary_file)

def main():
    
    filepaths=[
        ("en","./es-en/europarl-v7.es-en.en"),
        ("es","./es-en/europarl-v7.es-en.es"),
        ("de","./de-en/europarl-v7.de-en.de"),
        ("fr","./fr-en/europarl-v7.fr-en.fr"),
        ("it","./it-en/europarl-v7.it-en.it"),
        ("pl","./pl-en/europarl-v7.pl-en.pl")
    ]


    train=[]
    tf_idf_dics={}

    if GENERATECORPUS:
        #clean
        clean_dics={}
        for id,filepath in filepaths:
            if not os.path.isfile(filepath):
                print("File path {} does not exist. Exiting...".format(filepath))
                sys.exit()

            bag_of_words = {}
            with open(filepath) as fp:
                cnt = 0
                for line in fp:
                    record_word_cnt(normalize(line), bag_of_words)
                    cnt += 1
                    if cnt==cleaning: break
            bag_of_words={a:b for a,b in bag_of_words.items()}
            sorted_words = order_bag_of_words(bag_of_words, desc=True)
            print("Most frequent 100 words {}".format(sorted_words[:100]))
            clean_dics[id]=set((w[0] for w in sorted_words[:100]))

        # combines all words
        all=set([])
        for c in clean_dics.values():
            all=all.union(c)
        # saves words from each dictionary
        # remaining set must be filtered out later
        clean_dics2={id:all.difference(c) for id,c in clean_dics.items()}
        
        for id,filepath in filepaths:
            if not os.path.isfile(filepath):
                print("File path {} does not exist. Exiting...".format(filepath))
                sys.exit()

            bag_of_words = {}
            total=0
            with open(filepath) as fp:
                cnt = 0
                for line in fp:
                    if cnt<cleaning:
                        pass
                    else:
                        words=normalize(line)
                        skip=any(w in clean_dics2[id] for w in words)
                        if skip:
                            cnt-=1
                        else:
                            if random.random()*11<1:
                                train.append((id,line))
                            else:
                                total+=record_word_cnt(words, bag_of_words)
                    cnt += 1
                    if cnt==cleaning+training+test: break
            # tf:
            bag_of_words={a:b/total for a,b in bag_of_words.items()}
            tf_idf_dics[id]=bag_of_words
        for _,base in tf_idf_dics.items():
            for w in base:
                number=0
                for _,other in tf_idf_dics.items():
                    if other.get(w)!=None:
                        number+=1
                idf=len(tf_idf_dics)/number
                base[w]*=idf #calculate tf-idf
            sorted_words = order_bag_of_words(base, desc=True)
            save(id,bag_of_words)
            #bag_of_words2=load(id)
            print()
        save("train",train)
    else:
        tf_idf_dics={id:load(id) for id,_ in filepaths}
        train=load("train")

    contingency={id:{"tp":0,"fn":0,"tn":0,"fp":0} for id,_ in filepaths}
    confusion={id:{id:0 for id,_ in filepaths} for id,_ in filepaths}
    
    for t in train:
        id,line=t
        words=normalize(line)
        #winner_id
        winner_total=0
        for id2,dic in tf_idf_dics.items():
            # calculate for each word
            total=0
            for w in words:
                r=dic.get(w)
                if r!=None:
                    total+=r
            # totals of tf-idf
            if total>winner_total:
                winner_id=id2
                winner_total=total
        confusion[id][winner_id]+=1
        if id == winner_id:
            contingency[id]["tp"]+=1
            for id2,_ in tf_idf_dics.items(): 
                if id2!=id:
                    contingency[id]["fn"]+=1
        else:
            contingency[id]["tn"]+=1
            contingency[winner_id]["fp"]+=1
            for id2,_ in tf_idf_dics.items(): 
                if id2!=id and id2!=winner_id:
                    contingency[id]["fn"]+=1
            # confusion matrix       
    print("Contingency",contingency)
    print("Confusion",confusion)
    print()
    
def order_bag_of_words(bag_of_words, desc=False):
   words = [(word, cnt) for word, cnt in bag_of_words.items()]
   return sorted(words, key=lambda x: x[1], reverse=desc)

def record_word_cnt(words, bag_of_words):
    subtotal=0
    for word in words:
        if word != '':
            if word.lower() in bag_of_words:
                bag_of_words[word.lower()] += 1
            else:
                bag_of_words[word.lower()] = 1
            subtotal+=1
    return subtotal

if __name__ == "__main__":
    main()