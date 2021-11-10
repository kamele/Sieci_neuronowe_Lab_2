import MLP_v2_pomocnicze as pomocnicze
import MLP_v2

def test():
    dane_treningowe = pomocnicze.wczytajDane("train-images.idx3-ubyte")
    klasy_treningowe = pomocnicze.wczytajKlasy("train-labels.idx1-ubyte")
    klasy_treningowe = pomocnicze.zmien_klase_na_neurony(klasy_treningowe, 10)
    dane_treningowe=dane_treningowe/255
    print("Happy")


if __name__ == '__main__':
    test()