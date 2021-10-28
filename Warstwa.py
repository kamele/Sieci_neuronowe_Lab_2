import numpy as np
import idx2numpy


def MLP(liczba_warstw, dane, klasy, liczby_neuronow, funkcje_aktywacji, pochodne_funkcji, liczba_epok, wspolczynnik_uczenia, wielkosc_bacha):
    # dane_dla_warstwy = dane
    warstwy = []
    dane_dla_warstw=[]
    dane_dla_warstw.append(dane)

    #tworzenie warstw
    for i in range(liczba_warstw):
        warstwa = Warstwa(liczby_neuronow[i], funkcje_aktywacji[i],pochodne_funkcji[i], wspolczynnik_uczenia);
        if i == 0:
            # print(len(dane))
            warstwa.losujWagiWarstwy(len(dane[0]))
            # print(warstwa.wagi)
        else:
            warstwa.losujWagiWarstwy(liczby_neuronow[i - 1])
        warstwy.append(warstwa)

    for e in range(liczba_epok):
        print("===========================")
        print("epoka ", e)

        start_index = 0
        end_index = wielkosc_bacha
        if (end_index > len(dane) and start_index < len(dane)):
            end_index = len(dane) - 1
        print("w")
        while end_index<len(dane):
            dane_dla_warstwy = []
            dane_dla_warstw.append(dane[start_index:end_index])
            print("l", len(dane_dla_warstw))

            # obliczanie wartości
            for i in range(liczba_warstw):
                # print(i)
                dane_dla_warstwy = warstwy[i].wylicz(dane_dla_warstw[len(dane_dla_warstw) - 1])
                dane_dla_warstw.append(dane_dla_warstwy)
            # print("===========================")

            # Propagacja bledu
            # warstwa koncowa
            warstwy[len(warstwy) - 1].bladKoncowaWarstwa(klasy)
            warstwy[len(warstwy) - 1].aktualizujWagi(wielkosc_bacha, warstwy[len(warstwy) - 2].wyjscie)
            warstwy[len(warstwy) - 1].aktualizujBias(wielkosc_bacha)

            # pozostale warstwy
            for b in range(liczba_warstw - 2, -1, -1):
                # print(b)
                warstwy[b].obliczPochodnaFunkcjiAktywacji()
                warstwy[b].bladWarstwa(warstwy[b + 1].bladWarstwy, warstwy[b + 1].wagi)
                warstwy[b].aktualizujWagi(wielkosc_bacha, warstwy[b - 1].wyjscie)
                warstwy[b].aktualizujBias(wielkosc_bacha)

            #indeksy bacha
            start_index=end_index+1
            end_index = end_index + wielkosc_bacha
            if (end_index>len(dane) and start_index<len(dane)):
                end_index=len(dane)-1



        print(warstwy[len(warstwy)-1].wyjscie)
        # blad sredniokwadratowy
        print(np.sum((klasy-warstwy[len(warstwy)-1].wyjscie)**2) )#/liczby_neuronow[len(liczby_neuronow)-1])

    return warstwy





class Warstwa:
    liczba_neuronow=0
    wagi=[]
    funkcjaAktywacji=None
    pochodnaFunkcjaAktywacji=None
    pobudzenie = []
    wyjscie=[]
    pochodna_aktywacji=[]
    bladWarstwy = []

    def __init__(self, liczba_neuronow, funkcjaAktywacji, pochodnaFunkcjaAktywacji, wspolczynnik_uczenia):
        self.liczba_neuronow=liczba_neuronow
        self.funkcjaAktywacji=funkcjaAktywacji
        self.pochodnaFunkcjaAktywacji=pochodnaFunkcjaAktywacji
        self.bias = np.random.normal(0, 1, 1)
        self.wspolczynnik_uczenia=wspolczynnik_uczenia

    def losujWagiWarstwy(self, liczba_wymiarow_poprzedniej):
        self.wagi = losujWagi(liczba_wymiarow_poprzedniej, self.liczba_neuronow)
        print("wagi", np.shape(self.wagi))

    def wylicz(self,dane):
        # print(self.wagi)
        # print(np.shape(dane))
        self.pobudzenie=self.bias+np.dot(dane, self.wagi)
        # print(self.pobudzenie)
        self.wyjscie=self.funkcjaAktywacji(self.pobudzenie)
        # print(np.shape(self.wyjscie)," ",type(self.wyjscie))
        return self.wyjscie

    def obliczPochodnaFunkcjiAktywacji(self):
        # print("p")
        # print(self.pobudzenie)
        self.pochodna_aktywacji = self.pochodnaFunkcjaAktywacji(self.pobudzenie)
        # print(self.pochodna_funkcji_aktywacji)

    def bladKoncowaWarstwa(self,klasy):
        # print(klasy)
        # print(self.funkcjaAktywacji(self.pobudzenie))
        self.bladWarstwy=klasy-self.funkcjaAktywacji(self.pobudzenie)
        return self.bladWarstwy

    def bladWarstwa(self, blad_warstwy_nastepnej, wagi_warstwy_nastepnej):
        # print(np.shape(blad_warstwy_nastepnej))
        # print(np.shape(wagi_warstwy_nastepnej))
        # print(np.shape(self.pochodna_aktywacji))
        self.bladWarstwy=np.dot(blad_warstwy_nastepnej,wagi_warstwy_nastepnej.T)*self.pochodna_aktywacji
        return self.bladWarstwy

    def aktualizujWagi(self, wielkosc_batcha,wyjscie_poprzedniej_warstwy):
        # print("wagi")
        # print(np.shape(self.wagi)," - sum(",np.shape(wyjscie_poprzedniej_warstwy.T)," * ",np.shape(self.bladWarstwy),")")
        # print(np.shape(self.bladWarstwy))
        # print(np.shape(wyjscie_poprzedniej_warstwy))
        self.wagi=self.wagi-(self.wspolczynnik_uczenia/wielkosc_batcha)*np.sum(np.dot(wyjscie_poprzedniej_warstwy.T,self.bladWarstwy))
        return self.wagi

    def aktualizujBias(self, wielkosc_batcha):
        self.bias=self.bias-(self.wspolczynnik_uczenia/wielkosc_batcha)*np.sum(self.bladWarstwy)
        return self.bias

def zmien_klase_na_neurony(klasy, liczba_neuronow):
    nowe_klasy=np.zeros((len(klasy),liczba_neuronow))
    for i in range(len(klasy)):
        nowe_klasy[i][klasy[i]]=1

    return nowe_klasy

def test():
    dane_treningowe = wczytajDane("train-images.idx3-ubyte")
    klasy_treningowe = wczytajKlasy("train-labels.idx1-ubyte")
    klasy_treningowe = zmien_klase_na_neurony(klasy_treningowe, 10)
    #
    # warstwa_wejściowa= Warstwa(2, Warstwa.sigmoidalna)
    # warstwa_wejściowa.wylicz(dane_treningowe,klasy_treningowe)
    liczby_neuronow = [5, 3, 10]
    funkcje_aktywacji = [tanh, tanh, softmax]
    pochodne_funkcje_aktywacji = [pochodna_tanh, pochodna_tanh, pochodna_softmax]
    print(len(klasy_treningowe[2:100]))
    model = MLP(3, dane_treningowe[2:15],klasy_treningowe[2:15], liczby_neuronow, funkcje_aktywacji,pochodne_funkcje_aktywacji, 8, 0.5, 13)

    dane_dla_warstw=dane_treningowe[505:507]
    klasy_dla_warstw = klasy_treningowe[505:507]
    for warstwa in model:
            dane_dla_warstw = warstwa.wylicz(dane_dla_warstw)

    print("=========================================================")
    print("Wynik")
    print("-----------------------------")
    blad = (klasy_dla_warstw - model[len(model) - 1].wyjscie) ** 2
    # print(blad[0])
    # print(np.around(blad[0], decimals=2))
    # print(blad[1])
    # print(np.around(blad[1], decimals=2))
    print(np.sum((klasy_dla_warstw - model[len(model) - 1].wyjscie) ** 2))
    print("-----------------------------")
    # print(klasy_dla_warstw - model[len(model) - 1].wyjscie)
    # print(np.around(model[len(model) - 1].wyjscie, decimals=1))

    # print((np.around(model[len(model) - 1].wyjscie, decimals=2)).max(1))
    print(np.argmax(np.around(model[len(model) - 1].wyjscie, decimals=2),axis=1))
    # print(dane_dla_warstw)
    # print(np.sum(dane_dla_warstw[0]))
    # print(np.sum(dane_dla_warstw[1]))
    # print(klasy_treningowe[505:507])
    print(np.argmax(klasy_treningowe[505:507],axis=1))



#pomocnicze
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
    wagi=np.random.normal( 0, 1,(liczba_wag, liczba_neuronow))
    # print("Lwagi ",len(wagi))
    return wagi

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

def MLP_manual(liczba_warstw):
    dane_treningowe = wczytajDane("train-images.idx3-ubyte")
    klasy_treningowe = wczytajKlasy("train-labels.idx1-ubyte")
    # print(dane_treningowe[1:3])

    warstwa_wejściowa = Warstwa(2, tanh)
    dane_po_warstwie = warstwa_wejściowa.wylicz(dane_treningowe[1:3])

    warstwa_2 = Warstwa(3, ReLu)
    dane_po_warstwie2 = warstwa_2.wylicz(dane_po_warstwie)

    warstwa_wyjściowa = Warstwa(10, softmax)
    dane_po_warstwie_wyj = warstwa_wyjściowa.wylicz(dane_po_warstwie2)

    # wynik = softmax(warstwa_wyjściowa.pobudzenie)
    # print(wynik)
    # print(np.sum(wynik[0]))
    # print(np.sum(wynik[1]))
    print(dane_po_warstwie_wyj)


if __name__ == '__main__':
    test()