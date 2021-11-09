import numpy as np
import idx2numpy


#funkcje pomocnicze
def zmien_klase_na_neurony(klasy, liczba_neuronow):
    nowe_klasy=np.zeros((len(klasy),liczba_neuronow))
    for i in range(len(klasy)):
        nowe_klasy[i][klasy[i]]=1
    return nowe_klasy

def wczytajDane(nazwa_pliku):
    dane = idx2numpy.convert_from_file(nazwa_pliku)
    # print(dane.shape)
    dane = np.reshape(dane,(dane.shape[0],(dane.shape[1]*dane.shape[2])))
    # print(dane.shape)
    return dane

def wczytajKlasy(nazwa_pliku):
    dane = idx2numpy.convert_from_file(nazwa_pliku)
    # print(dane.shape)
    return dane

def losujWagi(liczba_wag, liczba_neuronow):
    wagi=np.random.normal( 0.0, 0.01,(liczba_wag, liczba_neuronow))
    # print("Lwagi ",len(wagi))
    return wagi


#funkcje aktywacji
def sigmoidalna(z):
    wynik=1/(1+np.exp(-z))
    return wynik

def tanh(z):
    # wynik=(2/(1+np.exp(-2*z)))-1
    wynik = np.tanh(z)
    return wynik

def pochodna_tanh(z):
    return (1-tanh(z)**2)

def ReLu(z):
    wynik=np.maximum(0,z)
    return wynik

def softmax(z):
    wynik = []
    for i in range(len(z)):
        w=softmax_helper(z[i])
        # print(w)
        wynik.append(w)
        # print(wynik)
    array=np.array(wynik)
    return array

def softmax_helper(z):
    e_d_elem = np.exp(z)
    suma_e_d_elem = np.sum(e_d_elem)
    # print(e_d_elem/suma_e_d_elem)
    return e_d_elem/suma_e_d_elem

def pochodna_softmax(z):
    print("Użyto pochodnej softmax")
    return 0