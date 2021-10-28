from abc import ABC, abstractmethod
import numpy as np

class Funkcja_aktywacji(ABC):
    @abstractmethod
    def funkcja_aktywacji(self,z):
        pass

    @abstractmethod
    def pochodna_funkcji(self, z):
        pass


class Softmax(Funkcja_aktywacji):
    def funkcja_aktywacji(self, z):
        wynik = []
        for i in range(len(z)):
            w = self.softmax_helper(z[i])
            # print(w)
            wynik.append(w)
            # print(wynik)
        return wynik

    def softmax_helper(self, z):
        e_d_elem = np.exp(z)
        suma_e_d_elem = np.sum(e_d_elem)
        # print(e_d_elem/suma_e_d_elem)
        return e_d_elem / suma_e_d_elem

class ReLu(Funkcja_aktywacji):
    def funkcja_aktywacji(self, z):
        wynik = np.maximum(0, z)
        return wynik

class Tanh(Funkcja_aktywacji):
    def funkcja_aktywacji(self, z):
        # wynik=(2/(1+np.exp(-2*z)))-1
        wynik = np.tanh(z)
        return wynik

class Sigmoidalna(Funkcja_aktywacji):
    def funkcja_aktywacji(self, z):
        wynik = 1 / (1 + np.exp(-z))
        return wynik
