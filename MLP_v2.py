import numpy as np
import MLP_v2_pomocnicze as pomocnicze

def MLP(liczba_warstw, dane, klasy, liczby_neuronow, funkcje_aktywacji, pochodne_funkcji,
        liczba_epok, wspolczynnik_uczenia, wielkosc_bacha, dane_testowe, klasy_testowe):
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



