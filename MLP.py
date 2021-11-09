import numpy as np
import idx2numpy


def MLP(liczba_warstw, dane, klasy, liczby_neuronow, funkcje_aktywacji, pochodne_funkcji, liczba_epok, wspolczynnik_uczenia, wielkosc_bacha, dane_testowe, klasy_testowe):
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
        # print(warstwa.wagi)
        warstwy.append(warstwa)

    for e in range(liczba_epok):

        print("===========================")
        print("epoka ", e)

        index_start=0
        index_end=wielkosc_bacha
        if(index_end>len(dane))and index_start<len(dane)-1:
            index_end=len(dane)-1

        while index_end<len(dane):
            # print("bach (",index_start," , ",index_end)
            # dane_dla_warstw = []
            # dane_dla_warstw.append(dane)
            dane_dla_warstw=dane[index_start:index_end]

            #obliczanie wartości
            for i in range(liczba_warstw):
                print(i)
                dane_dla_warstw = warstwy[i].wylicz(dane_dla_warstw)
                # dane_dla_warstw.append(dane_dla_warstwy)
            # print("===========================")

            # Propagacja bledu
            # warstwa koncowa
            warstwy[len(warstwy)-1].bladKoncowaWarstwa(klasy[index_start:index_end])

            # pozostale warstwy
            for b in range(liczba_warstw-2, -1, -1):
                # print("b",b)
                warstwy[b].obliczPochodnaFunkcjiAktywacji()
                warstwy[b].bladWarstwa(warstwy[b+1].bladWarstwy,warstwy[b+1].wagi)


            for b in range(liczba_warstw-1, 0, -1):
                # print("b",b)
                warstwy[b].aktualizujWagi((index_end-index_start), warstwy[b-1].wyjscie)
                warstwy[b].aktualizujBias((index_end-index_start))

            warstwy[0].aktualizujWagi((index_end-index_start), dane[index_start:index_end])
            warstwy[0].aktualizujBias((index_end-index_start))

            index_start=index_end+1
            index_end+=wielkosc_bacha
            if(index_end>len(dane)):
                if index_start<len(dane)-1:
                    index_end=len(dane)-1



        # print(warstwy[len(warstwy)-1].wyjscie)
        # blad sredniokwadratowy
        # print(np.shape(klasy))
        # print(np.shape(warstwy[len(warstwy)-1].wyjscie))
        # print(np.sum((klasy-warstwy[len(warstwy)-1].wyjscie)**2) /(liczby_neuronow[len(liczby_neuronow)-1]))

        dane_warstwy = dane_testowe
        klasy_warstwy = klasy_testowe
        # print("dane",np.sum(dane_warstwy[50:70], axis=1))
        for warstwa in warstwy:
            dane_warstwy = warstwa.wylicz(dane_warstwy)
            # print(warstwa.wagi)
            # print("dane", np.sum(dane_warstwy[50:70], axis=1))

        print(policzTrafnosc(dane_warstwy,klasy_warstwy))
        print(np.argmax(dane_warstwy[50:70], axis=1))
        print(np.argmax(klasy_warstwy[50:70], axis=1))

        print(np.sum((klasy_warstwy[50:70]-dane_warstwy[50:70])**2) /(liczby_neuronow[len(liczby_neuronow)-1]))

        # print(dane_warstwy[50:70])


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
        self.bias = 0 #np.random.normal(0, 1, 1)
        self.wspolczynnik_uczenia=wspolczynnik_uczenia

    def losujWagiWarstwy(self, liczba_wymiarow_poprzedniej):
        self.wagi = losujWagi(liczba_wymiarow_poprzedniej, self.liczba_neuronow)
        print("wagi", np.shape(self.wagi))

    def wylicz(self,dane):
        # print(self.wagi)
        # print(np.shape(dane))
        self.pobudzenie=self.bias+np.dot(dane, self.wagi)
        # print("dane",np.sum(dane, axis=1))
        # print("wagi", self.wagi)
        # print("pobudzenie",self.pobudzenie)
        self.wyjscie=self.funkcjaAktywacji(self.pobudzenie)
        # print("wyjscie",self.wyjscie)
        # print(np.shape(self.wyjscie)," ",type(self.wyjscie))
        return self.wyjscie

    def obliczPochodnaFunkcjiAktywacji(self):
        # print("p")
        # print(self.pobudzenie)
        self.pochodna_aktywacji = self.pochodnaFunkcjaAktywacji(self.pobudzenie)
        # print(self.pochodna_funkcji_aktywacji)

    def bladKoncowaWarstwa(self,klasy):
        # print(np.shape(klasy))
        # print(np.shape(self.funkcjaAktywacji(self.pobudzenie)))
        self.bladWarstwy=self.funkcjaAktywacji(self.pobudzenie)-klasy
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
        # self.wagi=self.wagi-(self.wspolczynnik_uczenia/wielkosc_batcha)*np.dot(self.bladWarstwy.T,wyjscie_poprzedniej_warstwy)
        self.wagi = self.wagi - (self.wspolczynnik_uczenia / wielkosc_batcha) * \
                    np.sum(np.dot(wyjscie_poprzedniej_warstwy.T, self.bladWarstwy))
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
    dane_treningowe=dane_treningowe/255
    # print(dane_treningowe[0])
    #
    # warstwa_wejściowa= Warstwa(2, Warstwa.sigmoidalna)
    # warstwa_wejściowa.wylicz(dane_treningowe,klasy_treningowe)
    liczby_neuronow = [20, 10]
    funkcje_aktywacji = [tanh,  softmax]
    pochodne_funkcje_aktywacji = [pochodna_tanh, pochodna_softmax]
    # print(len(klasy_treningowe[2:100]))
    model = MLP(2, dane_treningowe[2:305],klasy_treningowe[2:305], liczby_neuronow, funkcje_aktywacji, pochodne_funkcje_aktywacji, 10, 0.1, 303, dane_treningowe[2:305],klasy_treningowe[2:305])

    # dane_dla_warstw=dane_treningowe[505:507]
    # klasy_dla_warstw = klasy_treningowe[505:507]
    # for warstwa in model:
    #         dane_dla_warstw = warstwa.wylicz(dane_dla_warstw)

    print("=========================================================")
    print("Wynik")
    print("-----------------------------")
    # blad = (klasy_dla_warstw - model[len(model) - 1].wyjscie) ** 2
    # print(blad[0])
    # print(np.around(blad[0], decimals=2))
    # print(blad[1])
    # print(np.around(blad[1], decimals=2))
    # print(np.sum((klasy_dla_warstw - model[len(model) - 1].wyjscie) ** 2))
    print("-----------------------------")
    # print(klasy_dla_warstw - model[len(model) - 1].wyjscie)
    # print(np.around(model[len(model) - 1].wyjscie, decimals=1))

    # print((np.around(model[len(model) - 1].wyjscie, decimals=2)).max(1))
    # print(np.argmax(np.around(model[len(model) - 1].wyjscie, decimals=2),axis=1))
    # print(dane_dla_warstw)
    # print(np.sum(dane_dla_warstw[0]))
    # print(np.sum(dane_dla_warstw[1]))
    # print(klasy_treningowe[505:507])
    # print(np.argmax(klasy_treningowe[505:507],axis=1))



#pomocnicze
def policzTrafnosc(wyjscie, klasy):
    wyjscie_dec = np.argmax(wyjscie, axis=1)
    klasy_dec = np.argmax(klasy, axis=1)
    suma_poprawnych=0
    for i in range(len(wyjscie)):
        if wyjscie_dec[i]==klasy_dec[i]:
            # print("wyjscie_dec=" + str(wyjscie_dec[i]) + " == klasy_dec= " + str(klasy_dec[i]))
            suma_poprawnych+=1
    print("suma_poprawnych="+str(suma_poprawnych)+" / liczba= "+str(len(wyjscie)))
    return suma_poprawnych/len(wyjscie)

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