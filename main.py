import pandas as pd
from sklearn.datasets import load_iris
from sklearn.feature_selection import VarianceThreshold
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score,davies_bouldin_score

def lecture_fichier(nom):
    df = pd.read_csv(nom)
    return df

df=lecture_fichier('avril2919.csv')
df['date'] = pd.to_datetime(df['date'],infer_datetime_format=True)
df.set_index('date',inplace=True)
df['portdst']=df['portdst'].apply(str)

def entropy(frame):
    probs = pd.value_counts(frame.values.flatten(), normalize=True)
    entropy = -1 * np.sum(np.log2(probs) * probs)
    return (entropy)

def vecteur_entropy(df):
    test = df.groupby(['ipsrc'])['ipdst'].apply(entropy)
    h = []
    for i in range(0, np.size(test)):
        h.append(test[i])
    return (h)

def list_port(df):
    listt = []
    for name, group in df.groupby(['ipsrc'])['portdst']:
        u = group.tolist()
        listt.append(u)
    return(listt)

def list_ipdest(df):
    listtt = []
    for name, group in df.groupby(['ipsrc'])['ipdst']:
        v = group.tolist()
        listtt.append(v)
    return (listtt)

def matrice(df):
    t_dst = df.groupby(['ipsrc'])['ipdst'].apply(entropy)
    t_pdst = df.groupby(['ipsrc'])['portdst'].apply(entropy)
    array_ips = []
    array_ips = t_dst.index.values

    array_ipdst = []
    for i in range(0, np.size(t_dst)):
        array_ipdst.append(t_dst[i])

    array_pdst = []
    for i in range(0, np.size(t_pdst)):
        array_pdst.append(t_pdst[i])

    v = np.array([array_ips, array_ipdst, array_pdst, list_ipdest(df), list_port(df)])
    v = np.transpose(v)
    return (v)

def ecriture(W):
    import csv
    with open('matrice.csv', 'w', newline='') as f:  # Ouverture du fichier CSV en écriture
        ecrire = csv.writer(f)  # préparation à l'écriture
        ecrire.writerow(['ipsource','entropy ipdest','entropy portdst','ipdest','portdest'])
        for i in W:  # Pour chaque ligne de la matrice...
            ecrire.writerow(i)  # Mettre dans la variable ecrire la nouvelle ligne
    print('', end='\n')
    print('longueur du tableau : ', len(W))
    f.close()

def ecriture_cl(W):
    import csv
    with open('Cluster.csv', 'w', newline='') as f:  # Ouverture du fichier CSV en écriture
        ecrire = csv.writer(f)  # préparation à l'écriture
        ecrire.writerow(['ipsource','entropy ipdest','entropy portdst','ipdest','portdest','Num de cluster'])
        for i in W:  # Pour chaque ligne de la matrice...
            ecrire.writerow(i)  # Mettre dans la variable ecrire la nouvelle ligne
    print('', end='\n')
    print('longueur du tableau : ', len(W))
    f.close()

def elbow(X):
    inertia = []
    k_range = range(1, 50)
    for k in k_range:
        model = KMeans(n_clusters=k).fit(X)
        inertia.append(model.inertia_)

    return(k_range, inertia)


def kmean(k,X):
    model = KMeans(n_clusters=k)
    model.fit(X)
    f = model.predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=f)
    b = model.cluster_centers_  # obtenir la position finale de cluster
    plt.scatter(b[:, 0], b[:, 1], c='r')

def cluster(k,X,v):
    model = KMeans(n_clusters=k)
    model.fit(X)
    f = model.predict(X)
    h = np.array([v[:, 0], v[:, 1], v[:, 2], v[:, 3], v[:, 4], f])
    h = np.transpose(h)
    return(h)

def generer(X,u):
    db, slc = {}, {}
    for k in range(2, u):
        kmeans = KMeans(n_clusters=k, max_iter=1000, random_state=10).fit(X)
        if k == 3: labels = kmeans.labels_
        clusters = kmeans.labels_
        db[k] = davies_bouldin_score(X, clusters)
        slc[k] = silhouette_score(X, clusters)
    return(db,slc)

def attack(h):
    l, c = h.shape
    a = max(h[:, 5])

    ls0 = []
    ls1 = []
    ls2 = []
    ls3 = []
    ls4 = []
    ls5 = []
    ls6 = []
    ez = []
    bz = []
    for i in range(0, a + 1):
        s1 = [1 for t in range(0, l) if h[t, 5] == i]
        s2 = [h[k, 1] for k in range(0, l) if h[k, 5] == i]
        s3 = [h[k, 2] for k in range(0, l) if h[k, 5] == i]
        s4 = [np.size(h[k, 3]) for k in range(0, l) if h[k, 5] == i]
        s5 = [np.size(h[k, 4]) for k in range(0, l) if h[k, 5] == i]
        s6 = [np.unique(h[k, 3]).size for k in range(0, l) if h[k, 5] == i]
        ls0.append(i)
        ls1.append(np.sum(s1))
        ls2.append(round(np.mean(s2), 3))
        ls3.append(round(np.mean(s3), 3))
        ls4.append(np.sum(s4))
        ls5.append(np.sum(s5))
        ls6.append(max(s6))
        # Scan vertical et horizontal
        if (ls6[i] >= 29):
            ez.append("attaque DDoS")
        elif (ls2[i] < ls3[i]):
            ez.append("vertical")
        elif (ls2[i] > ls3[i]):
            ez.append("horizontal")
        elif (ls2[i] == ls3[i] == 0):
            ez.append("défaut de configuration")
        else:
            ez.append("sous réseau")
        # Pourcentage d'elements dans le cluster
        c = (ls1[i] * 100) / l
        bz.append(round(c, 2))

    p = np.array([ls0, ls1, ls2, ls3, ls4, ls5, ls6, ez, bz])
    p = np.transpose(p)
    dff = pd.DataFrame(
        {'num cluster': p[:, 0], 'nbr ipsrc': p[:, 1], 'entropy ipdst': p[:, 2], 'entropy portdst': p[:, 3],
         'nbr ipdst': p[:, 4], 'nbr portdst': p[:, 5], 'nbr ipdst sans redondances': p[:, 6], 'type de scan': p[:, 7],
         '% elements': p[:, 8]})

    return (dff)

def graphe_attack(dff):
    dff['entropy ipdst'] = pd.to_numeric(dff['entropy ipdst'])
    dff['entropy portdst'] = pd.to_numeric(dff['entropy portdst'])

    dff1 = dff.loc[dff['type de scan'] == 'attaque DDoS']

    plt.figure(figsize=(8, 6), dpi=90)
    plt.scatter(dff['entropy ipdst'], dff['entropy portdst'], c='g', marker='+', linewidth=3, alpha=0.9)
    plt.scatter(dff1['entropy ipdst'], dff1['entropy portdst'], c='r', marker='+', linewidth=3)
    plt.annotate('Attaque DDoS', xy=(3, 1), xytext=(4, 2), c='r', size=15)
    plt.annotate('Normal', xy=(3, 1), xytext=(1.5, 2.7), c='g', size=20)
    plt.grid()
    plt.xlim(-1, 6)
    plt.xlabel('Entropy ipdst')
    plt.ylabel('Entropy portdst')
    plt.title('Graphe des attaques DDoS')