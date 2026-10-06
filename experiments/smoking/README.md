# Eksperyment 3D: FlyGym / MuJoCo + spiking connectome

Wybrany symulator to FlyGym 1.2.1 z MuJoCo 3.2.7. Ciało, chód, kontakt nóg
z podłożem i rendering są trójwymiarowe. Arena jest płaska; nie modelujemy lotu.
Obiekt `cigarette` jest abstrakcyjnym markerem źródła zapachu, bez animacji palenia.

## Instalacja (z katalogu głównego projektu)

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
mkdir -p vendor data
git clone https://github.com/dtch1997/fly-api.git vendor/fly-api
git -C vendor/fly-api checkout a6ad07a810b1a43cd0356149c07b32105eb46d2a
git clone https://github.com/philshiu/Drosophila_brain_model.git vendor/Drosophila_brain_model
git -C vendor/Drosophila_brain_model checkout 91bdd1e7dcf193f3e7ca5a8933497fcef63b7960
curl -fL https://raw.githubusercontent.com/flyconnectome/flywire_annotations/v2.1.0/supplemental_files/Supplemental_file1_neuron_annotations.tsv -o data/annotations.tsv
```

`requirements.lock.txt` zapisuje wszystkie wersje z działającego lokalnego
środowiska Python 3.12/Linux; dla odtworzenia tego środowiska można użyć go zamiast
`requirements.txt`.

Repozytoria i dane zostały pobrane lokalnie w tej sesji. Polecenia klonowania są
dla nowego checkoutu projektu. Dane, środowisko i wyniki są ignorowane przez Git.

## Uruchamianie

```bash
.venv/bin/python experiments/smoking/smoking_experiment.py
```

Na Linuxie bez okna można ustawić `MUJOCO_GL=egl`. Nagrania powstają podczas
symulacji i są zapisywane jako MP4, a nie wyświetlane w interaktywnym GUI.
`--no-video` wyłącza rendering. Szybkie sprawdzenie całego mechanizmu:

```bash
MUJOCO_GL=egl .venv/bin/python experiments/smoking/smoking_experiment.py --quick --contact-start --out results/contact_check
```

`--contact-start` ustawia początek epizodu obok źródła wyłącznie jako diagnostykę
kontaktu/plastyczności. Taki przebieg nie jest dowodem nauczenia nawigacji.
Domyślnie wszystkie warunki startują dalej od źródła. `--decisions`,
`--train-episodes`, `--test-episodes` i `--seed` pozwalają zmienić długość i ziarno.

Test zapisanych wag bez uczenia:

```bash
MUJOCO_GL=egl .venv/bin/python experiments/smoking/smoking_experiment.py --test-only --weights-dir results/smoking --out results/test_only
```

Użyj tego samego `--seed`, danych modelu, adnotacji i parametrów obwodu co przy
treningu. Test nie ma nicotine, zewnętrznego pobudzenia PAM ani aktualizacji wag.
Endogenne spike'i PAM nadal mogą występować. Program sprawdza niezmienność wag
podczas PRE i TEST.

## Mechanizm i źródła kodu

- `vendor/fly-api/experiments/navigation/nav_demo.py`: rzeczywista klasa `Brain`,
  wybór ORN/KC/MBON/PAM, PoissonGroup, `_episode`, SpikeMonitor i sterowanie ciałem.
- `vendor/fly-api/experiments/learning/model_ext.py`: `build_subnet`, wagi Brian2
  `syn.w` w woltach, znak i liczba połączeń z connectomu.
- `vendor/fly-api/experiments/learning/learning_driver_mb.py`: wzór na LTD aktywnych
  KC→MBON z bramką opartą na zmierzonej aktywności PAM.
- `vendor/Drosophila_brain_model/model.py`: parametry LIF Shiu.
- `config.py`: wszystkie parametry eksperymentalne, funkcja `nicotine_to_pam`.
- `smoking_experiment.py`: scena 3D, kontakt, dynamika nicotine/tolerance,
  pobudzenie PAM, aktualizacja `brain.syn.w[brain.plastic_pos]`, A/B i wyniki.

Źródło tabeli: https://github.com/flyconnectome/flywire_annotations (v2.1.0,
materializacja 783). Brakujące etykiety ORN w `cell_type` uzupełniamy istniejącymi
etykietami `hemibrain_type`; przetworzona tabela zostaje w wynikach. Nie wymyślamy
identyfikatorów neuronów. W tej tabeli liczebności obwodu mogą się różnić od demo.

Kontakt jest sprawdzany na podstawie odległości XY, co jest odpowiednie dla
płaskiej areny. W TRAIN zwiększa nicotine i tolerance. W A nicotine podaje
częstotliwości Poissona na rzeczywiste neurony PAM. W B wymuszone PAM wynosi zero.
Wagi zmieniają się wyłącznie w A/TRAIN, gdy zmierzona aktywność PAM przekroczy
bramkę; depresja obejmuje synapsy z presynaptycznych KC, które faktycznie spikowały.
Kontrola B ma takie samo nicotine/tolerance, sensorykę, stan początkowy mózgu,
ziarna symulatora i eksplorację. PRE dostarcza pomiaru przed treningiem.

Sterowanie używa różnicy MBON z dwóch wirtualnych czułków plus identycznego
szumu eksploracyjnego. Nie ma nagrody RL ani reguły skrętu na podstawie położenia
obiektu. Jest to jednak ręcznie zaprojektowany adapter MBON→chód z demo autora,
nie pełne sterowanie przez anatomiczny obwód neuronów zstępujących. Nie przejmujemy
reguły zatrzymania w źródle używanej przez upstream demo.

## Wyniki i interpretacja

Domyślnie `results/smoking/` zawiera:

- `smoking_experiment.csv`: dane każdej decyzji, fazy, odległości, czas kontaktu,
  pierwszy kontakt (puste = nie osiągnięto), nicotine, tolerance, PAM,
  średnią wagę, aktywność MBON/PAM i prędkość zbliżania;
- `summary.png`: sześć wykresów;
- `A_weights.npy`, `B_weights.npy`, `initial_weights.npy`, `plastic_pre.npy`;
- `config.json`, `provenance.json`, `summary.json`, `annotations_resolved.tsv`
  i nagrania `A_TRAIN_*.mp4` itd.

Porównuj zmianę PRE→TEST w A i B, szczególnie średni dystans, czas przy źródle
i odsetek epizodów z kontaktem. Puste czasy pierwszego kontaktu trzeba traktować
jako nieosiągnięcie, nie zero. Wyniki jednego ziarna nie dowodzą efektu; powtórz
dla wielu ziaren. Jeśli TRAIN nie zawiera kontaktu lub PAM nie przekracza bramki,
nie będzie uczenia. Nie gwarantujemy powstania preferencji.

## Uproszczenia i parametry

Parametry nicotine, tolerancji, PAM, próg kontaktu, stałe zapachu, eksploracja,
learning rate oraz czas decyzji są arbitralnymi parametrami symulacji.
Intake i tolerance są przyrostami na sekundę kontaktu, decay na krok decyzji;
nicotine/tolerance resetujemy w każdym epizodzie, pamięć synaptyczna pozostaje.
Zapach to sztuczne gaussowskie pole z wirtualnymi czułkami wyznaczanymi z pozycji
ciała. Oba źródła mają disjoint zestawy sześciu klas ORN. Marker nie ma kolizji.
Węch nie symuluje transportu gazu ani dymu.

Neurony i początkowe połączenia są z rzeczywistego connectomu, parametry LIF
z modelu Shiu. Dziedziczymy cztery modyfikacje stabilizujące obwód z fly-api:
wyzerowane szybkie wyjścia DAN, KC→KC, wejścia do ORN i dodatnie wyjścia ALLN.
Gain KC→MBON ×20 jest parametrem adaptera odczytu w demo, nie pomiarem biologicznym.
LTD jest jednorodne, bez podziału na przedziały grzybkowatego ciała.
Dwa sniffs po 150 ms mózgu odpowiadają jednej decyzji 150 ms ciała: to rozdzielone
zegary demonstratora, a nie model skalibrowany w czasie biologicznym.
Warunek C z losowym PAM pozostaje poza minimalną wersją A/B.
