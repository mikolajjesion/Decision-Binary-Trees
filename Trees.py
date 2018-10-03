import numpy as np
import time

# tab = [2,7,5,2,6,None,9,None,None,5,11,None,None,4]

S = [[0, 1, 0, 1, 0, 1],[0, 0, 0, 1, 0, 0],[0, 0, 0, 0, 1, 0],[1, 0, 1, 0, 1, 0],[0, 1, 1, 1, 0, 1],[0, 1, 0, 0, 1, 1],[1, 1, 1, 0, 0, 0],[1, 1, 1, 1, 0, 1],[0, 1, 1, 0, 1, 0],[1, 1, 0, 0, 0, 1],[1, 0, 0, 0, 1, 0]]
test = [[0, 0, 1, 1, 0, 0],
[1, 1, 0, 1, 1, 0],
[0, 1, 1, 1, 0, 1],
[1, 0, 1, 0, 0, 1],
[1, 0, 0, 0, 0, 0],
[0, 0, 1, 0, 0, 1],
[0, 1, 1, 1, 1, 1],
[1, 0, 0, 1, 0, 0],
[1, 1, 1, 1, 1, 0],
[1, 0, 1, 1, 0, 0],
[1, 1, 1, 0, 1, 1],
[1, 0, 0, 0, 1, 0],
[0, 0, 1, 1, 1, 0],
[0, 1, 0, 0, 1, 1],
[1, 0, 1, 1, 1, 0],
[0, 1, 0, 1, 0, 1],
[0, 0, 0, 1, 1, 0],
[0, 0, 0, 1, 0, 0],
[1, 1, 1, 1, 0, 1],
[1, 1, 0, 0, 1, 1],
[1, 1, 0, 1, 0, 0],
[0, 1, 0, 0, 0, 0]]

def p0(X):
    """

    :param X: Macierz danych wejściowych binarnych
    :return: Funkcja zwraca prawdopodobieństwo etykiet y=0 nad wszystkimi etykietami
    """
    N = len(X)
    if N == 0: return 0
    p0 = []
    return len([p0.append(X[i][-1:]) for i in range(N) if X[i][-1:] == [0]])/N


def p1(X):
    """

    :param X: Macierz danych wejściowych binarnych
    :return: Funkcja zwraca prawdopodobieństwo etykiet y=1 nad wszystkimi etykietami
    """
    N = len(X)
    if N == 0: return 0
    p_1=[]
    return len([p_1.append(X[i][-1:]) for i in range(N) if X[i][-1:] == [1]])/N

def H(X):
    """

    :param X: Macierz danych wejściowych
    :return: Funkcja zwraca wartość entropii dla danych binarnych z macierzy X
    """
    P0 = p0(X)
    P1 = p1(X)
    output = 0
    if P0 != 0:
        output += -P0*np.log2(P0)
    if P1 != 0:
        output += - P1*np.log2(P1)
    return output

def X_i_u(S,i,u):
    """

    :param S: Macierz danych binarnych
    :param i: kolumna w macierzy S
    :param u: wartość binarna (0,1)
    :return: Funkcja zwraca podzbiór zbioru macierzy S, dzieli macierz S na dwie części
    """
    X_iu = []
    [X_iu.append(S[row]) for row in range(len(S)) if S[row][i] == u]
    return X_iu

def IG(X,i):
    """

    :param X: Macierz danych binarnych
    :param i: kolumna macierzy X
    :return: Funkcja zwraca miarę "zysku informacji" (information gain)
    """
    N = len(X)
    q_i_0 = len(X_i_u(X,i,0))/N
    q_i_1 = len(X_i_u(X,i,1))/N
    return H(X) - q_i_0 * H(X_i_u(X,i,0)) - q_i_1 * H(X_i_u(X,i,1))

def ID3(S, Tree = [], ind = 1):
    """

    :param S: Macierz danych binarnych
    :param Tree: Drzewo binarne
    :param ind: Indeks w drzewie binarnym
    :return: Funkcja zwraca drzewo binarne
    """
    if type(S) != np.array:
        S = np.array(S)

    argIG = []
    suma = np.sum(S[:, -1])

    if suma == len(S):
        Tree[ind-1] = (S[0, -1], 1)
        return Tree
    if suma == 0:
        Tree[ind-1] = (S[0, -1], 1)
        return Tree

    for i in range(len(S[1,:])-1):
        argIG.append(IG(S,i))

    max_arg = argIG.index(max(argIG))
    Tree[ind-1] = (max_arg, 0)

    Tree = ID3(X_i_u(S,max_arg,0), Tree, 2 * ind)
    Tree = ID3(X_i_u(S,max_arg,1), Tree, (2 * ind)+1)
    return Tree

def read_txt(file):
    """

    :param file: nazwa pliku z danymi
    :return: Funkcja zwraca macierz danych z pliku
    """
    M = open(file,'r').read().split('\n')
    M = [M[i].split(',') for i in range(len(M))]
    M = np.array(M)
    temp = np.matrix(M[:,0])
    M = np.delete(M,0,axis=1)
    M = np.append(M,temp.transpose(),axis=1)
    M = np.array(M)
    return M

txt = 'mushrooms.txt'
def letters_in_columns(data,col=0,Dict = {}):
    """

    :param data: Macierz danych z pliku
    :param col: numer kolumny w macierzy data
    :param Dict: słownik
    :return: Funkcja zwraca słownik, gdzie kluczami są numery kolumn a wartościami unikatowe wartości z każdej z kolumn
    """
    Dict[col] = []
    for i in range(len(data[:,col])-1):
        if data[i,col] not in Dict[col]:
            Dict[col].append(data[i,col])
    if col < len(data[1,:])-1:
        return letters_in_columns(data,col+1,Dict)
    else:
        return Dict

def one_Hot(Dict):
    """

    :param Dict: Słownik z danymi o unikatowych wartościach z każdej z kolumn macierzy wejściowej
    :return: Funkcja zwraca słownik, gdzie kluczami są krotki
    (unikatowe wartości z każdej z kolumn macierzy wejściowej, kolumna występowania)
    oraz wartościami jest lista reprezentacji binarnych unikatowych wartości.
    """
    dict = {}
    tab = []
    ind = 0
    for key in range(len(Dict)):
        for i in range(len(Dict[key])):
            if len(Dict[key]) == 2:
                dict[(Dict[key][0],key)] = [1]
                dict[(Dict[key][1], key)] = [0]
                break
            else:
                tab.append([0] * len(Dict[key]))
                tab[ind][i] = 1
                dict[(Dict[key][i],key)] = tab[ind][:]
                ind+=1
    return dict

def convert_txt_into_Binary_Data(Data):
    """

    :param Data: Macierz danych wejściowych
    :return: Funkcja zwraca binarną reprezentację danych wejściowych w postaci listy list
    """
    Binary = one_Hot(letters_in_columns(read_txt(txt)))
    tab = []
    output = []
    for row in range(len(Data[:,1])-1):
        for i in range(len(Data[row,:])):
            tab = tab + Binary[(Data[row,i],i)]
        output.append(tab)
        tab = []

    return output

def walk_tree(tree, x, i=0, dec=0):
    """

    :param tree: drzewo binarne
    :param x: wektor wartości binarnych testowych
    :param i: indeks w drzewie binarnym
    :param dec: index
    :return: Funkcja zwraca krotkę (wartości drzewa pod indeksem i-tym, etykietę wektora x, index)
    """
    if tree[i][1] == 0:
        return walk_tree(tree, x, 2 * i + 1 + x[tree[i][0]], dec + 1)
    else:
        return (tree[i][0], x[-1], dec + 1)

def Tree_test(tree, test):
    """

    :param tree: drzewo binarne
    :param test: dane testowe
    :return: Funkcja liczy dokładność drzewa do danych testowych i zwraca krotkę
    (ilość wierszy w danych testowych, ilość błędów , dokładność drzewa)
    """
    len_test = len(test)
    len_error = 0

    for x in test:
        rec, exp, _ = walk_tree(tree, x)
        if rec != exp:
            len_error += 1

    return (len_test, len_error, (len_test - len_error) / len_test)

def trening_test_podz(data, scale):
    """

    :param data: Dane testowe
    :param scale: skala podziału danych
    :return: Funkcja zwraca krotkę (dane treningowe, dane testowe)
    """
    point_of_division = int(len(data) * scale)
    return (data[:point_of_division], data[point_of_division:])

def to_one_hot(l, a):
    """

    :param l: długość wektora
    :param a: punkt wstawienia wartości do wektora l
    :return: Funkcja zwraca wektor zer gdzie pod indeksem a stoi 1, inaczej mowiac reprezentuje binarnie dane
    """
    return list(np.eye(l)[a])

def Tree_cut(tree,id = 1):
    """

    :param tree: Drzewo binarne
    :param id: długośc wektora binarnej reprezentacji danych
    :return: Funkcja zwraca poddrzewa wygenerowane z drzewa wejściowego
    """
    trees = []
    for key in tree.keys():
        if tree[key][1] == 0:
            if id == 1:
                child1 = tree.copy()
                child2 = tree.copy()
                child1[key] = (1, 1)
                child2[key] = (0, 1)
                trees.append(child1)
                trees.append(child2)
            else:
                for i in range(id):
                    child = tree.copy()
                    child[key] = (to_one_hot(id, i),1)
                    for j in child:
                        trees.append(child)
    return trees

def del_orph(tree,id = 1, dict = {}):
    """

    :param tree: Drzewo binarne
    :param id: indeks w drzewie
    :param dict: słownik wyjściowy
    :return: Funkcja zwraca słownik poddrzewa binarnego {indeks: (wartość , korzeń/liść)}
    """
    if tree[id-1][1] == 0:
        dict[id-1] = tree[id-1]
        dict = del_orph(tree, 2*id,dict)
        dict = del_orph(tree, (2*id)+1 , dict)
    elif tree[id-1][1] == 1:
        dict[id-1] = tree[id-1]

    return dict

def Trees_clear(trees):
    """

    :param trees: poddrzewa binarne
    :return: Funkcja zwraca listę słowników zawierających reprezentację poddrzew binarnych
    """
    cleared = []
    for i in trees:
        cleared.append(del_orph(i).copy())
    return cleared

def Leaf_Count(tree):
    """

    :param tree: Drzewo binarne
    :return: Funkcja zwraca ilość liści w danym drzewie
    """
    leafs = 0
    for i in tree.keys():
        if tree[i][1] == 1:
            leafs +=1
    return leafs

def accuracy_data1(trees,tree,test,accuracy):
    """

    :param trees: poddrzewa binarne
    :param tree: drzewo binarne
    :param test: dane testowe
    :param accuracy: dokładność danego drzewa
    :return: Funkcja wypisuje najlepsze poddrzewo, alfę , dokładność
    """
    tab = []
    LT = Leaf_Count(tree)
    for i in trees:
        tab.append(100 * ((accuracy - Tree_test(i, test)[2]) / (LT - Leaf_Count(i))))
    best_tree = trees[tab.index(min(tab))]
    Acc = Tree_test(best_tree,test)[2]
    print("najlepsze poddrzewo to: " + str(best_tree) + " - ({:.2f}% alfa poddrzewa)".format(min(tab))+ " : ({:.2f}% dokładność)".format(100*Acc))

def accuracy_mushrroms(mushroom_tree,M_test,md,trees_muhsrooms):
    """

    :param mushroom_tree: Drzewo binarne
    :param M_test: Dane testowe wydzielone
    :param md: dokładność dla danego drzewa
    :param trees_muhsrooms: poddrzewa binarne
    :return: Funkcja wypisuje najlepsze poddrzewo, alfę , dokładność
    """
    tab = []
    ML = Leaf_Count(mushroom_tree)
    for i in trees_muhsrooms:
        tab.append(((md-Tree_test(i,M_test)[2])/(ML-Leaf_Count(i)))*100)
    alfa = min(tab)
    best_tree = trees_muhsrooms[tab.index(alfa)]
    Acc = Tree_test(best_tree,M_test)[2]
    print("najlepsze poddrzewo to: " + str(best_tree) + " - ({:.2f}% alfa poddrzewa)".format(min(tab)) + " : ({:.2f}% dokładność)".format(100*Acc))


"""
Test dokładności drzewa na danych S
"""
tree = ID3(S,{},1)
len_test, len_error,accuracy = Tree_test(tree,test)
print("Drzewo Dane1")
print(tree)
print((len_test - len_error), "/", len_test, "({:.2f}% dokładności do danych testowych)".format(100*accuracy))
print()

"""
Dzielenie drzewa z danych S na poddrzewa
"""
prunned_trees = Tree_cut(tree)
trees = Trees_clear(prunned_trees)
print("poddrzewa data1: "+str(trees))

"""
Wybranie najlepszego poddrzewa dla danych1
"""
accuracy_data1(trees,tree,test,accuracy)

"""
Test dokładności drzewa na danych mushrooms
"""
print("_______________________________________________________________________________________________")
print()
mushroom_bin = convert_txt_into_Binary_Data(read_txt(txt))
M_train, M_test = trening_test_podz(mushroom_bin, 0.7)
mushroom_tree = ID3(M_train,{},1)
print("Drzewo Mushrooms")
print(mushroom_tree)
mlt, mle, md = Tree_test(mushroom_tree, M_test)
mlt2, mle2, md2 = Tree_test(mushroom_tree, M_train)

print((mlt - mle), "/", mlt, "({:.2f}% dokładności na wydzielonym zbiorze mushrooms (testowy))".format(100*md))
print((mlt2 - mle2), "/", mlt2, "({:.2f}% dokładności na wydzielonym zbiorze mushrooms (treningowy))".format(100*md2))

"""
Dzielenie drzewa z danych mushrroms na poddrzewa
"""
prunned_mushrooms = Tree_cut(mushroom_tree)
trees_muhsrooms = Trees_clear(prunned_mushrooms)
print("poddrzewa mushrooms: "+str(trees_muhsrooms))

"""
Wybranie najlepszego poddrzewa w mushrooms
"""
accuracy_mushrroms(mushroom_tree,M_test,md,trees_muhsrooms)
