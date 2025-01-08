# Brygida Silawko i Tomasz Smoleń Zadanie 5 Raport

Nr. albumu 331 434

### 1. Decyzje projektowe

Parametryzowalne wartości sieci to:

- funkcje aktywacji i aktywacji wyjścia

- pochodne powyższych funkcji oraz funkcji straty

- sposób inicjowania wag i biasów

- rozmiary warstw

- przekształcanie targetu

  Do sieci podajemy obiekt pd.DataFrame z wraz z klasą celową.

W funkcji train możemy też wybrać:

- ilość epok

- rozmiar mini batcha

- learning rate

Input skalujemy, żeby wartości wektora mieścił się w wartości pomiędzy 0 a 1.

Wybieramy najbardziej prawdopodobną klasę, którą zwrócił softmax.

Cyfry z MNIST kodujemy one hot encoding.

## 2. Cel eksperymentów

Celem było zaimplementowanie perceptronu wielowarstwowego oraz wybranego algorytmu optymalizacji gradientowej z algorytmem propagacji wstecznej, a potem go wyuczyć do klasyfikacji zbioru danych MNIST.

Badamy wpływ `mini_batch_size` i `learning_rate`  na działanie perceptronu na zbiorze walidacyjnym, po czym puszczamy go jeszcze raz na zbiorze testowym

## 3. Wyniki i omówienie eksperymentów

Dane:

> data/test.csv

Wykres:

> data/results_plot.png

Badane sizes: 5, 10, 30, 50, 75, 100

Badane learning rates: 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9

Romiary warstw:  64,  20, 16, 10

Obliczamy średnią z 20 pomiarów.

Korzystamy z funkcji aktywacyjnej ReLu, ostatnia warstwa ma funkcje Softmax.

Funkcja straty to funkcja średnio-kwadratowa.

Wagi inicjalizujemy losowo w przedziale latex tutaj

Biasy inicjalizujemy losowo w przedziale od -1 do 1

Ilość epok: 20

![chart.png](\\wsl.localhost\Ubuntu\home\tomek\wsi\wsi-lab-brygida\lab5\data\results_plot.png)

Po wykresie widać, że dla `mini_batch_size` 10 perceptron radzi sobie najlepiej.

Najwyższa średnia wyszła dla learning rate 0.8, aczkolwiek od 0.3 różnice sa marginalne.

Ogólnie też wyniki były logarytmicznie lepsze dla coraż wyższych learning rate.

Dla większych batch size perceptron osiągał gorsze wyniki, dochodząc do 70 - kilku procent dla batch size 100.

Dla batch size 10 i learning rate 0.8 dokładność na zbiorze testowym wynosiła 96,29%.

Batch size 5 na początku sobie radzić najlepiej, ale po czasie learning rate był dla niego zbyt duży.

# 4. Wnioski

Percpetron wielowarstwowy bardzo dobrze radzi sobie z odgdywaniem cyfr z zestawu MNIST o ile sie go wcześniej wytrenuje przy odpowiednich parametrach.  Dla zbyt dużych batch size gradient descent wykonywał zbyt mało kroków, co skutkowało w mniejszej dokładności, ponadtwo wyższy learning rate pozwalał algorytmowi efektywniej dopasować się do danych.
