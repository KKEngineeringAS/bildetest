import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np

# Funksjon for å laste opp bilde
def last_opp_bilde():
    filbane = filedialog.askopenfilename()
    if filbane:
        # Åpne bildet og vis det i GUI
        bilde = Image.open(filbane)
        bilde.thumbnail((300, 300))  # Endre størrelse på bildet for visning
        img_tk = ImageTk.PhotoImage(bilde)
        label_bilde.config(image=img_tk)
        label_bilde.image = img_tk
        
        # Kall funksjonen for å analysere bildet
        analyser_bilde(filbane)

# Funksjon for å analysere bildet og gjenkjenne kategorier
def analyser_bilde(filbane):
    # Last inn MobileNet-modellen som er forhåndstrent på ImageNet-datasettet
    modell = tf.keras.applications.mobilenet_v2.MobileNetV2(weights='imagenet')

    # Last inn og prosesser bildet
    bilde = Image.open(filbane)
    bilde = bilde.resize((224, 224))  # MobileNet krever 224x224 px bilder
    bilde_array = np.array(bilde)
    bilde_array = np.expand_dims(bilde_array, axis=0)  # Legg til batch-dimensjon
    bilde_array = tf.keras.applications.mobilenet_v2.preprocess_input(bilde_array)

    # Kjør bildet gjennom modellen for å få prediksjoner
    prediksjoner = modell.predict(bilde_array)

    # Dekodere resultatene for å finne kategorien
    resultat = tf.keras.applications.mobilenet_v2.decode_predictions(prediksjoner, top=1)[0]
    kategori = resultat[0][1]  # Dette gir kategorinavnet (som 'car', 'cat' osv.)
    
    # Vis resultatet i GUI
    label_tekst.config(text=f"Gjenkjent kategori: {kategori}")

# Opprett hovedvinduet
root = tk.Tk()
root.title("Bildegjenkjenning")

# Knapp for å laste opp bilde
knapp_lastopp = Button(root, text="Last opp bilde", command=last_opp_bilde)
knapp_lastopp.pack()

# Etikett for å vise bildet
label_bilde = Label(root)
label_bilde.pack()

# Etikett for å vise gjenkjent kategori
label_tekst = Label(root, text="Ingen kategori gjenkjent enda")
label_tekst.pack()

# Start GUI-løkken
root.mainloop()

