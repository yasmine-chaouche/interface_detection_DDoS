from tkinter import*
import pandas as pd
import main
from tkinter import filedialog,messagebox,ttk
import matplotlib
from sklearn.cluster import KMeans
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import numpy as np
import tkinter



def importer_data():
    root = Tk()
    root.geometry("1000x800")
    root.pack_propagate(False)
    # root.resizable(0,0)

    # frame for padding view
    frame1 = LabelFrame(root, text="Data Frame")
    frame1.place(height=550, width=1200)

    # frame for open file
    file_frame = LabelFrame(root, text="Ouvrir un fichier dataset")
    file_frame.place(height=100, width=400, rely=0.84, relx=0.2)

    # Buttons
    button1 =Button(file_frame, text="Browse a file", command=lambda: File_dialog())
    button1.place(rely=0.65, relx=0.50)

    button2 = Button(file_frame, text="Load file", command=lambda: load_data())
    button2.place(rely=0.65, relx=0.30)

    label_file = ttk.Label(file_frame, text='Pas de fichier selectionné')
    label_file.place(rely=0, relx=0)

    # widget
    tv1 = ttk.Treeview(frame1)
    tv1.place(relheight=1, relwidth=1)

    treescrolly =Scrollbar(frame1, orient="vertical", command=tv1.yview)
    treescrollx = Scrollbar(frame1, orient="horizontal", command=tv1.xview)
    tv1.configure(xscrollcommand=treescrollx.set, yscrollcommand=treescrolly.set)
    treescrollx.pack(side="bottom", fill="x")
    treescrolly.pack(side="right", fill="y")

    def File_dialog():
        filename = filedialog.askopenfilename(initialdir="/",
                                              title="selectionner un fichier",
                                              filetype=(
                                              ("csv files", "*.csv"), ("xlsx files", "*.xlsx"), ("All Files", "*.*")))
        label_file["text"] = filename
        return None

    def load_data():
        file_path = label_file["text"]
        try:
            csv_filename = r"{}".format(file_path)
            df = main.lecture_fichier(csv_filename)
        except ValueError:
            messagebox.showerror("Information", "Le fichier choisi est invalide")
            return None
        except FileNotFoundError:
            messagebox.showerror("Information", f"fichier nexicte pas avec le chemin{file_path}")
            return None

        clear_data()
        tv1["column"] = list(df.columns)
        tv1["show"] = "headings"
        for column in tv1["columns"]:
            tv1.heading(column, text=column)

        df_rows = df.to_numpy().tolist()
        for row in df_rows:
            tv1.insert("", "end", values=row)
        return (df)


    def clear_data():
        tv1.delete(*tv1.get_children())
        tv1.delete(*tv1.get_children())

def elbow():
    root = tkinter.Tk()
    root.wm_title("Graphe Elbow")

    fig = Figure(figsize=(5, 4), dpi=100)

    df = main.lecture_fichier('avril2919.csv')
    v = main.matrice(df)
    X = v[:, 1:3]
    a, b = main.elbow(X)
    fig.add_subplot(111).plot(a, b)
    fig.add_subplot(111).set_title('Elbow')
    fig.add_subplot(111).set_xlabel('nbr de clusters')

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

    tkinter.mainloop()

def silhouette():
    root = tkinter.Tk()
    root.wm_title("Graphe Silhouette")

    fig = Figure(figsize=(5, 4), dpi=100)

    df = main.lecture_fichier('avril2919.csv')
    v = main.matrice(df)
    X = v[:, 1:3]
    db, slc = main.generer(X, 80)
    fig.add_subplot(111).plot(list(db.keys()), list(db.values()))
    fig.add_subplot(111).set_title('sihlouette')

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

    tkinter.mainloop()

def cluster():
    df = main.lecture_fichier('avril2919.csv')
    v = main.matrice(df)
    main.ecriture(v)
    root = tkinter.Tk()
    root.wm_title("Graphe de Cluster")

    fig = Figure(figsize=(5, 4), dpi=100)

    df = main.lecture_fichier('avril2919.csv')
    v = main.matrice(df)
    X = v[:, 1:3]
    model = KMeans(n_clusters=8)
    model.fit(X)
    f = model.predict(X)
    fig.add_subplot(111).scatter(X[:, 0], X[:, 1], c=f)
    b = model.cluster_centers_
    fig.add_subplot(111).scatter(b[:, 0], b[:, 1], c='r')

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

    tkinter.mainloop()

def bouldin():
    root = tkinter.Tk()
    root.wm_title("Graphe Elbow")

    fig = Figure(figsize=(5, 4), dpi=100)

    df = main.lecture_fichier('avril2919.csv')
    v = main.matrice(df)
    X = v[:, 1:3]
    db, slc = main.generer(X, 80)
    fig.add_subplot(111).plot(list(slc.keys()), list(slc.values()))
    fig.add_subplot(111).set_title('Bouldin')

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

    tkinter.mainloop()

def calc_entropy():
    df = main.lecture_fichier('avril2919.csv')
    v = main.matrice(df)
    main.ecriture(v)
    root = Tk()
    root.geometry("1000x800")
    root.pack_propagate(False)
    # root.resizable(0,0)

    # frame for padding view
    frame1 = LabelFrame(root, text="Data Frame")
    frame1.place(height=550, width=1200)
    df = main.lecture_fichier('matrice.csv')
    # widget
    tv1 = ttk.Treeview(frame1)
    tv1.place(relheight=1, relwidth=1)
    tv1["column"] = list(df.columns)
    tv1["show"] = "headings"
    for column in tv1["columns"]:
        tv1.heading(column, text=column)

    df_rows = df.to_numpy().tolist()
    for row in df_rows:
        tv1.insert("", "end", values=row)
    treescrolly = Scrollbar(frame1, orient="vertical", command=tv1.yview)
    treescrollx = Scrollbar(frame1, orient="horizontal", command=tv1.xview)
    tv1.configure(xscrollcommand=treescrollx.set, yscrollcommand=treescrolly.set)
    treescrollx.pack(side="bottom", fill="x")
    treescrolly.pack(side="right", fill="y")

    root.mainloop()

def affiher_cluster():
    df = main.lecture_fichier('avril2919.csv')
    v = main.matrice(df)
    X=v[:,1:3]
    h=main.cluster(80,X,v)
    main.ecriture_cl(h)
    root = Tk()
    root.geometry("1000x800")
    root.pack_propagate(False)
    # root.resizable(0,0)

    # frame for padding view
    frame1 = LabelFrame(root, text="Data Frame")
    frame1.place(height=550, width=1200)
    df = main.lecture_fichier('Cluster.csv')
    # widget
    tv1 = ttk.Treeview(frame1)
    tv1.place(relheight=1, relwidth=1)
    tv1["column"] = list(df.columns)
    tv1["show"] = "headings"
    for column in tv1["columns"]:
        tv1.heading(column, text=column)

    df_rows = df.to_numpy().tolist()
    for row in df_rows:
        tv1.insert("", "end", values=row)
    treescrolly = Scrollbar(frame1, orient="vertical", command=tv1.yview)
    treescrollx = Scrollbar(frame1, orient="horizontal", command=tv1.xview)
    tv1.configure(xscrollcommand=treescrollx.set, yscrollcommand=treescrolly.set)
    treescrollx.pack(side="bottom", fill="x")
    treescrolly.pack(side="right", fill="y")

    root.mainloop()

def data_attack():
    df = main.lecture_fichier('avril2919.csv')
    v = main.matrice(df)
    X=v[:,1:3]
    h=main.cluster(80,X,v)
    dff=main.attack(h)
    root = Tk()
    root.geometry("1000x800")
    root.pack_propagate(False)
    # root.resizable(0,0)

    # frame for padding view
    frame1 = LabelFrame(root, text="Data Frame")
    frame1.place(height=550, width=1200)
    # widget
    tv1 = ttk.Treeview(frame1)
    tv1.place(relheight=1, relwidth=1)
    tv1["column"] = list(dff.columns)
    tv1["show"] = "headings"
    for column in tv1["columns"]:
        tv1.heading(column, text=column)

    df_rows = dff.to_numpy().tolist()
    for row in df_rows:
        tv1.insert("", "end", values=row)
    treescrolly = Scrollbar(frame1, orient="vertical", command=tv1.yview)
    treescrollx = Scrollbar(frame1, orient="horizontal", command=tv1.xview)
    tv1.configure(xscrollcommand=treescrollx.set, yscrollcommand=treescrolly.set)
    treescrollx.pack(side="bottom", fill="x")
    treescrolly.pack(side="right", fill="y")

    root.mainloop()

def graphe_attack():
    df = main.lecture_fichier('avril2919.csv')
    v = main.matrice(df)
    X=v[:,1:3]
    h=main.cluster(80,X,v)
    dff=main.attack(h)

    root = tkinter.Tk()
    root.wm_title("Graphe des attaques")
    fig = Figure(figsize=(8, 6), dpi=90)

    dff['entropy ipdst'] = pd.to_numeric(dff['entropy ipdst'])
    dff['entropy portdst'] = pd.to_numeric(dff['entropy portdst'])

    dff1 = dff.loc[dff['type de scan'] == 'attaque DDoS']
    fig.add_subplot(111).scatter(dff['entropy ipdst'], dff['entropy portdst'], c='g', marker='+', linewidth=3, alpha=0.9)
    fig.add_subplot(111).scatter(dff1['entropy ipdst'], dff1['entropy portdst'], c='r', marker='+', linewidth=3)
    fig.add_subplot(111).annotate('Attaque DDoS', xy=(3, 1), xytext=(4, 2), c='r', size=15)
    fig.add_subplot(111).annotate('Normal', xy=(3, 1), xytext=(1.5, 2.7), c='g', size=20)
    fig.add_subplot(111).grid()
    #fig.add_subplot(111).xlim(-1, 6)
    fig.add_subplot(111).set_xlabel('Entropy ipdst')
    fig.add_subplot(111).set_ylabel('Entropy portdst')
    fig.add_subplot(111).set_title('Graphe des attaques DDoS')

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

    tkinter.mainloop()

root=Tk()

photo = PhotoImage(file="cerist.png")
canvas = Canvas(root,width=350, height=200)
canvas.create_image(0, 0, anchor=NW, image=photo)
canvas.pack(side=tkinter.LEFT)

champ_label = Label(root, text="Algorithme detection des Attaques DDoS d'un trafic réseau" )
champ_label.config(font=("Helvetica", 20),justify=CENTER, height=2)
champ_label.pack()



btn=Button(root,text="Ajouter Data",command=importer_data,height = 2 ,width =12, borderwidth=2,font=("Helvetica", 10))
btn.pack(padx=20,pady=10)
btn=Button(root,text="Calcule Entropy",command=calc_entropy,height = 2 ,width =12, borderwidth=2,font=("Helvetica", 10))
btn.pack(padx=20,pady=10)
btn=Button(root,text="Graphe de Cluster",command=cluster,height = 2 ,width =14, borderwidth=2,font=("Helvetica", 10))
btn.pack(padx=20,pady=10)
btn=Button(root,text="Afficher Cluster",command=affiher_cluster,height = 2 ,width =14, borderwidth=2,font=("Helvetica", 10))
btn.pack(padx=20,pady=10)
btn=Button(root,text="Graphe Elbow",command=elbow,height = 2 ,width =14, borderwidth=2,font=("Helvetica", 10))
btn.pack(padx=20,pady=10)
btn=Button(root,text="Graphe Silhouette",command=silhouette,height = 2 ,width =14, borderwidth=2,font=("Helvetica", 10))
btn.pack(padx=20,pady=10)
btn=Button(root,text="Graphe Bouldin",command=bouldin,height = 2 ,width =14, borderwidth=2,font=("Helvetica", 10))
btn.pack(padx=20,pady=10)
btn=Button(root,text="Afficher Attaques",command=data_attack,height = 2 ,width =14, borderwidth=2,font=("Helvetica", 10))
btn.pack(padx=20,pady=10)
btn=Button(root,text="Graphe Attaques",command=graphe_attack,height = 2 ,width =14, borderwidth=2,font=("Helvetica", 10))
btn.pack(padx=20,pady=10)


root.geometry("800x500")
root.title("Main window")
root.mainloop()