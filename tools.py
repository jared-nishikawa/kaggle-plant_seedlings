
labels = ["Black-grass", "Common_Chickweed", "Loose_Silky-bent", \
        "Shepherds_Purse", "Charlock", "Common_wheat", "Maize", \
        "Small-flowered_Cranesbill", "Cleavers", \
        "Fat_Hen", "Scentless_Mayweed", "Sugar_beet"]

def categorical(label):
    categories = [0]*12
    ind = labels.index(label)
    categories[ind] = 1
    return categories

def categorize(values):
    V = list(values)
    M = max(V)
    ind = V.index(M)
    return labels[ind]


