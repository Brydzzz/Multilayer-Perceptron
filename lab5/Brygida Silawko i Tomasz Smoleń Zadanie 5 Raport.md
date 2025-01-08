# Brygida Silawko i Tomasz Smoleń Zadanie 5 Raport

Nr. albumu 331 434

## 1. Decyzje projektowe

Parametryzowalne wartości sieci to:

- funkcje aktywacji i aktywacji wyjścia

- pochodne powyższych funkcji oraz funkcji straty

- sposób inicjowania wag i biasów

- rozmiary warstw

- przekształcanie targetu

Do sieci podajemy obiekt `pd.DataFrame` zawierający dane oraz oczekiwane wyniki.

W funkcji train możemy też wybrać:

- ilość epok

- rozmiar mini batcha

- learning rate

Zbiór MNIST ładujemy za pomocą pakietu `scikit-learn` (`sklearn.datasets.load_digits()`) jako `pd.DataFrame`. Zbiór z pakietu jest spłaszczony do postaci wektorów.

Dane skalujemy, żeby wartości wektora mieścił się w wartości pomiędzy 0 a 1.

Wybieramy najbardziej prawdopodobną klasę, którą zwrócił softmax.

Cyfry z MNIST kodujemy one hot encoding.

## 2. Cel i opis eksperymentów

Celem było zaimplementowanie perceptronu wielowarstwowego oraz wybranego algorytmu optymalizacji gradientowej z algorytmem propagacji wstecznej, a potem go wyuczyć do klasyfikacji zbioru danych MNIST.

Badamy wpływ `mini_batch_size` i `learning_rate` na działanie perceptronu na zbiorze walidacyjnym, po czym najlepszą znalezioną kombinacje sprawdzamy na zbiorze testowym.

- Badane rozmiary: `5, 10, 30, 50, 75, 100`

- Badane learning rates: `0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9`

- Rozmiary warstw: `64, 20, 16, 10`

- Obliczamy średnią dokładność na zbiorze walidacyjnym z 20 pomiarów, ponieważ wagi i biasy są losowo inicjalizowane.

- Korzystamy z funkcji aktywacyjnej ReLu, ostatnia warstwa ma funkcje Softmax.

- Funkcja straty to funkcja średnio-kwadratowa.

- Wagi inicjalizujemy losowo w przedziale od $-\sqrt\frac{6}{nin}$ do $\sqrt\frac{6}{nin}$

- Biasy inicjalizujemy losowo w przedziale od $-1$ do $1$

- Ilość epok: 20

## 3. Wyniki i omówienie eksperymentów

**Dane:**

> lab5/data/test.csv

**Wykres:**

> lab5/data/results_plot.png

![Wykres z wynikami](.\data\results_plot.png)

Po wykresie widać, że dla `mini_batch_size = 10` perceptron osiąga największe dokładności.

Najwyższa średnia dokładność była dla `learning_rate = 0.8`, aczkolwiek od `0.3` różnice sa marginalne.

Ogólnie też wyniki były logarytmicznie lepsze dla coraz wyższych `learning_rate`.

Dla większych batch size perceptron osiągał gorsze wyniki, dochodząc do 70 - kilku procent dla batch size 100.

Dokładność najlepszej pary na zbiorze testowym, czyli `mini_batch_size = 10` i `learning_rate = 0.8` wyniosła 96,29%.

Dokładność `mini_batch_size = 5` dla większych `learning_rate` (tzn. `> 0.5`) była niższa niż dla mniejszych `learning_rate`

## 4. Wnioski

Perceptron wielowarstwowy bardzo dobrze radzi sobie z odgadywaniem cyfr z zestawu MNIST o ile zostanie wcześniej wytrenowany z odpowiednimi parametrami. Dla zbyt dużych `batch_size` gradient descent wykonywał zbyt mało kroków, co skutkowało mniejszą dokładnością, ponadto wyższy `learning_rate` pozwalał algorytmowi efektywniej dopasować się do danych.
