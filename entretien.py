#Exercice 1
# Fonction trouver_doublons(nombres) -> 
# renvoie la liste des entiers apparaissant au moins deux fois dans nombres, sans doublons dans la sortie et
# en respectant l'odre de la première apparition dans la liste initiale nombres
# temporelle O(nlogn) ; 

# [1,3,7,10,7] output : [7]
# [1,2,7,2,7] output : [2,7]
def trouver_doublons(nombres):
    occurences = {}
    sortie = []
    # clé : nombre ; valeur : nb d'occurences 
    for i in range(len(nombres)):
        occurences[nombres[i]] = 1+ occurences.get(nombres[i],0) # o(n)
    for key in occurences:
            if occurences[key] > 1:
                 sortie.append(key)
    return sortie
            
#nombres = [1,2,7,2,7] 
#output = [2,7]

nombres = [1,9,6,5]


print(trouver_doublons(nombres))

# set() 
