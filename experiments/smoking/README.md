# Eksperyment 3D: FlyGym / MuJoCo + spiking connectome

Wybrany symulator to FlyGym 1.2.1 z MuJoCo 3.2.7. Ciało, chód, kontakt nóg
z podłożem i rendering są trójwymiarowe. Arena jest płaska; nie modelujemy lotu.
W LIVE papieros jest abstrakcyjnym obiektem z podnoszeniem, noszeniem, żarem
i animacją dymu. Sekwencję czynności obsługuje jawny automat; uczenie preferencji
zapachu pozostaje w obwodzie konektomu.

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
W PRE i TEST wszystkie warunki startują dalej od źródła. `--decisions`,
`--train-episodes`, `--test-episodes` i `--seed` pozwalają zmienić długość i ziarno.

### Trening z okazją do skojarzenia

Domyślny `--training-protocol balanced_exposure` umieszcza muchę w TRAIN
naprzemiennie przy źródle cigarette i przy źródle kontrolnym. W pojedynczej
ekspozycji emituje zapach tylko to źródło, aby nie nagradzać mieszaniny obu
bodźców. Nie podajemy PAM ręcznie: kontakt z cigarette nadal musi zwiększyć
nicotine, a zmierzona aktywność PAM musi przekroczyć bramkę plastyczności.
Warunki A/B mają identyczne ekspozycje i pozycje; w B wymuszone PAM wynosi zero.
PRE i TEST startują z oryginalnej, odległej pozycji i oba źródła emitują zapach.
To kontrolowany protokół warunkowania, nie spontaniczne odkrycie podczas TRAIN.

`--training-decisions` określa długość pojedynczej ekspozycji (domyślnie 12),
niezależnie od długości PRE/TEST. `--training-protocol free` przywraca trening
od dalekiej pozycji z oboma zapachami; wtedy brak kontaktów może oznaczać brak
uczenia. Wagi z wcześniejszych przebiegów bez kontaktów nie zawierają skojarzenia;
uruchom nowy trening zamiast samego `--test-only` na takich wagach.

Kontakt jest teraz wykrywany przy każdym kroku fizyki (0.1 ms). Intake, tolerance,
decay i czas przy źródle są naliczane w tym kroku; następny sniff używa aktualnego
nicotine. Dzięki temu wejście i wyjście pomiędzy decyzjami nie zostaje przeoczone.
CSV podaje `pam_rate` użyty w sniffie i nicotine na końcu kroku decyzyjnego.

Adapter chodu zwalnia przy małej aktywności MBON. Przy bardzo silnym lokalnym
zapachu i osłabionym względem stanu niewyuczonego MBON uruchamia ogólny odruch zatrzymania/żerowania, jednakowy
dla obu zapachów i A/B. Kierunek skrętu nadal wynika tylko z różnicy MBON dwóch
czułków. Progi, mniejszy szum eksploracyjny i LTD 0.3 są jawnymi parametrami
demonstratora, nie biologicznie zweryfikowaną polityką.

Przed epizodami mierzymy referencyjną odpowiedź niewyuczonego obwodu na każdy
czysty zapach (bez PAM), po czym przywracamy stan neuronów i RNG. Odczyt
`relative_MBON_valence` dzieli aktualną odpowiedź dwóch czułków przez sumę
referencji ważonych ich stężeniami. Żerowanie wymaga wartości poniżej 0.8
i stężenia powyżej 0.97. Wejście w ten bodziec
jest wykrywane w każdym kroku fizyki, aby nie przeskoczyć obszaru między
decyzjami mózgu. Normalizacja jest przybliżonym adapterem odczytu,
nie dodatkową plastycznością ani pełnym modelem neuronów zstępujących;
chroni przed maskowaniem wyuczonego osłabienia przez drugi zapach.
Referencje zapisujemy w `readout_reference.json`.

Przy osłabionym MBON i silnym zapachu (>0.85) mucha najpierw zwalnia do
`approach_drive=0.3`. Zmniejsza to przeskakiwanie źródła i odrzut ciała po
zatrzymaniu. Żerowanie ma histerezę: utrzymuje się, dopóki stężenie jest >0.95
i względna awersja <1.0. Te same zasady dotyczą obu zapachów i obu warunków;
nie sprawdzamy nazwy obiektu, aby wybrać kierunek lub zatrzymanie.

### Widoczny chód i długość filmu

Domyślna kamera śledzi pozycję muchy i pokazuje ją z bliska pod stałym kątem;
podłoże przesuwa się, gdy mucha idzie. Minimapka w rogu pokazuje jej trasę,
aktualną pozycję i oba źródła również wtedy, gdy nie mieszczą się w zbliżeniu.
`--camera arena` daje widok z góry z automatycznym dopasowaniem położenia i
oddalenia kamery do muchy i obu źródeł. Nie jest to już nieruchomy kadr,
z którego mucha może wyjść.
Zmiana kamery nie wpływa na sterowanie ani preferencję.

Każdy nowy film pokazuje warunek (`A`/`B`), fazę (`PRE`/`TRAIN`/`TEST`), stan
napędu i dystans. `FEEDING - MOTOR OFF` oznacza wyłączony napęd chodu; drobne
ruchy nóg i ciała wynikające z fizyki mogą nadal występować. Zatrzymania po
uczeniu należy szukać w `A_TEST_*.mp4`, a nie w PRE lub kontroli B. Obok każdego
MP4 zapisujemy JSON z fazą, czasem przy obiekcie i końcowym dystansem.

W TEST film pokazuje przynajmniej 5 sekund symulacji po pierwszym kontakcie
(`--observe-after-arrival 5`). Jeżeli planowany epizod kończy się wcześniej,
program kontynuuje fizykę i mózg, przy zerowym nicotine/PAM i zamrożonych wagach.
Nie jest to powielanie ostatniej klatki ani wymuszenie pozostania przy obiekcie:
odejście również będzie pokazane. Przy prędkości odtwarzania 0.25 daje to co
najmniej 20 sekund filmu po kontakcie. `--observe-after-arrival 0` wyłącza dodatek;
`--quick` domyślnie też go wyłącza.

Dodatkowe decyzje mają w CSV `observation_tail=True` i są pomijane w podstawowych
wykresach oraz `summary.json`, aby porównanie A/B nadal miało planowany horyzont.
JSON obok filmu opisuje pełne nagranie, w tym dodatkowy czas obserwacji.

Czas filmu to `decisions × decision_sec / play_speed`, pomniejszony o około
0.02 s symulowanego czasu pomijanego na początku przez FlyGym. Domyślne
40 decyzji to 6 s symulacji i około 12 s filmu. Domyślnie używamy po jednym
epizodzie PRE/TEST i pięciu ekspozycji TRAIN; dla analizy wielu powtórzeń zwiększ
`--test-episodes`. `--quick` to tylko 0.3 s
symulacji i około 0.6 s filmu: ten tryb służy sprawdzeniu uruchomienia.

Jeden dłuższy film z zapisanych wag, bez nowego treningu:

```bash
MUJOCO_GL=egl .venv/bin/python experiments/smoking/smoking_experiment.py \
  --test-only --weights-dir results/smoking --conditions A \
  --test-episodes 1 --decisions 40 --play-speed 0.25 --out results/motion_preview
```

To 6 s rzeczywistego ruchu w symulacji, odtwarzane przez około 24 s. Dla dłuższej
obserwacji zwiększ `--decisions`; zmniejszenie `--play-speed` tylko spowalnia
odtwarzanie. Domyślnie nadal uruchamiamy oba warunki A/B; `--conditions` ogranicza
je dla podglądu. Zapisujemy długość przebytej drogi i prędkość do CSV, a postęp
co dziesięć decyzji pokazuje czas symulacji i czas obliczeń.

Wywołanie renderingu odbywa się w rytmie klatek filmu, zamiast przy każdym kroku
fizyki. Kroki fizyki i mózgu pozostają takie same.

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
Kontrola B ma te same reguły nicotine/tolerance, sensorykę, stan początkowy mózgu,
ziarna symulatora i eksplorację; stężenia mogą różnić się wskutek wyuczonego
zachowania i czasu kontaktu. PRE dostarcza pomiaru przed treningiem.

Sterowanie używa różnicy MBON z dwóch wirtualnych czułków plus identycznego
szumu eksploracyjnego. Nie ma nagrody RL ani reguły skrętu na podstawie położenia
obiektu. Jest to jednak ręcznie zaprojektowany adapter MBON→chód z demo autora,
nie pełne sterowanie przez anatomiczny obwód neuronów zstępujących. Reguła
żerowania jest oparta na niskim MBON i bardzo silnym zapachu, bez używania
odległości do wyboru kierunku skrętu.

## Wyniki i interpretacja

Domyślnie `results/smoking/` zawiera:

- `smoking_experiment.csv`: dane każdej decyzji, fazy, odległości, czas kontaktu,
  pierwszy kontakt (puste = nie osiągnięto), nicotine, tolerance, PAM,
  średnią wagę, aktywność MBON/PAM i prędkość zbliżania;
- `summary.png`: sześć wykresów;
- `A_weights.npy`, `B_weights.npy`, `initial_weights.npy`, `plastic_pre.npy`;
- `config.json`, `provenance.json`, `summary.json`, `annotations_resolved.tsv`
  i nagrania `A_TRAIN_*.mp4` itd.

Jeżeli folder wyników nie jest pusty, nowe uruchomienie zapisuje do osobnego
podfolderu `run_...` i wypisuje jego ścieżkę. Chroni to stare wyniki i zapobiega
mieszaniu krótkich filmów `--quick` z nowszym CSV. W `--weights-dir` podaj folder
konkretnego przebiegu, który zawiera pliki wag i `provenance.json`.

Porównuj zmianę PRE→TEST w A i B, szczególnie średni dystans, czas przy źródle
i odsetek epizodów z kontaktem. Puste czasy pierwszego kontaktu trzeba traktować
jako nieosiągnięcie, nie zero. Wyniki jednego ziarna nie dowodzą efektu; powtórz
dla wielu ziaren. Jeśli TRAIN nie zawiera kontaktu lub PAM nie przekracza bramki,
nie będzie uczenia. Nie gwarantujemy powstania preferencji.

## Uproszczenia i parametry

Parametry nicotine, tolerancji, PAM, próg kontaktu, stałe zapachu, eksploracja,
learning rate oraz czas decyzji są arbitralnymi parametrami symulacji.
Intake i tolerance są przyrostami na sekundę kontaktu, decay jest przeliczany
z kroku decyzji na każdy krok fizyki;
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

## Ciągłe szukanie i pobieranie nikotyny

`--live` uruchamia jeden ciągły przebieg bez resetowania ciała między szukaniem
a kontaktem. Oba źródła są obecne przez cały czas. Przy kontakcie z papierosem
rośnie nikotyna i tolerancja; w A aktywne są PAM i plastyczność KC→MBON.
Na filmie kontakt jest oznaczony `SMOKING - NICOTINE INTAKE`. To abstrakcyjny
model pobierania nikotyny, bez animacji dymu.

```bash
MUJOCO_GL=egl .venv/bin/python experiments/smoking/smoking_experiment.py \
  --live --conditions A --duration 30 --play-speed 0.5 \
  --weights-dir results/smoking/run_20261007_161611_485707 \
  --out results/live
```

Checkpoint daje wcześniejszą wyuczoną reakcję zapachową. Bez `--weights-dir`
mózg zaczyna naiwny: znalezienie i zatrzymanie nie są zagwarantowane.
`--duration` określa czas fizyki, nie czas obliczeń; 30 sekund przy tej prędkości
daje około minutę filmu. Po pierwszym kontakcie działa też
`--observe-after-arrival` (domyślnie 5 sekund). Tryb PRE/TRAIN/TEST pozostaje
dostępny do porównania kontrolnego; LIVE nie jest testem z zamrożonymi wagami.

## Podgląd podczas obliczeń

Dodaj `--preview`, następnie otwórz `http://127.0.0.1:8765/` w przeglądarce.
Strona pokazuje aktualną klatkę 3D z kamery i minimapę jeszcze przed zapisem MP4.
Na pierwszą klatkę trzeba poczekać na inicjalizację mózgu. Podgląd aktualizuje
się w tempie obliczeń, niekoniecznie w czasie rzeczywistym. Serwer działa
do zakończenia procesu; MP4 pozostaje zapisane. `--no-video --preview` pozwala
oglądać bez przechowywania klatek filmu. `--preview-port 0` wybiera wolny port
i wypisuje adres. W zdalnym VS Code przekieruj ten port w zakładce Ports.

## Zużywanie papierosa

W `--live` papieros ma skończony zapas: domyślnie 5 sekund faktycznego
pobierania nikotyny. `--cigarette-duration 3` ustawia 3 sekundy. Czas szukania
nie zużywa zapasu; odejście przerywa zużywanie, powrót wykorzystuje resztę.
Po wyczerpaniu marker szarzeje, zapach znika i kończy się dostarczanie nikotyny.
Odruch zatrzymania wygasa przy następnej decyzji mózgu, bez teleportowania
muchy ani kierowania jej do kolejnego celu. HUD pokazuje procent i pozostały
czas; CSV zapisuje zapas, stan wyczerpania i rzeczywisty czas pobierania.
Nikotyna już pobrana pozostaje w organizmie i dalej zanika zgodnie z modelem.

To parametr symulacyjny, nie biologicznie wyznaczona trwałość papierosa.
Zapas resetuje się dopiero przy nowym przebiegu. PRE/TRAIN/TEST zachowują
dotychczasowy protokół. Arena LIVE zawiera domyślnie pięć papierosów i jedno źródło
kontrolne; zużyte papierosy nie odrastają automatycznie.

## Kilka papierosów na arenie

LIVE ma domyślnie pięć papierosów w losowanych pozycjach (mm). Każdy ma niezależny zapas i znika z pola
zapachu po zużyciu. Czujniki odbierają najsilniejszą lokalną smugę zapachu
niewyczerpanych papierosów; kierunek ruchu nadal wynika z odczytu mózgu.
Nie jest wymuszane odwiedzanie wszystkich źródeł. Kamera arena uwzględnia
wszystkie obiekty, a minimapa oznacza je C1–C5 i szarzeje po wyczerpaniu.
HUD pokazuje zapas każdego papierosa. CSV zawiera niezależne zapasy
w `cigarette_supplies_json` oraz liczbę `depleted_cigarettes`.

Własne położenia zastępują listę domyślną, np.:

```bash
MUJOCO_GL=egl .venv/bin/python experiments/smoking/smoking_experiment.py \
  --live --preview --conditions A --duration 30 \
  --cigarette-position 14 7 --cigarette-position 28 10 \
  --cigarette-position 42 4 --cigarette-duration 3 \
  --weights-dir results/smoking/run_20261007_161611_485707 \
  --out results/live
```

## Nawigacja między kolejnymi źródłami

Od startu LIVE adapter ruchowy wykorzystuje względny odczyt MBON.
Przy wczytaniu wag mierzymy reakcję na sam zapach papierosa bez PAM
i przywracamy stan mózgu oraz RNG; później używamy odczytu podczas pobierania nikotyny do ustalenia atrakcyjności
zapachu (próg taki jak dla odruchu zatrzymania), a różnicę stężenia na antenach
do określenia skrętu. Sygnał kierunkowy nie jest mnożony przez malejącą liczbę
spike’ów MBON. Reakcja zapachowa
jest zachowana między kontaktami; nie zapamiętuje pozycji ani trasy.
To jawny odruch sensoryczny, nie wyuczona trasa ani kompletny neuronalny
model sterowania ruchem. Nie dostaje numeru celu ani kierunku do jego pozycji.
Dzięki temu depresja MBON nie zeruje sygnału kierunkowego po kontakcie.
Chód LIVE od startu jest ograniczony do wolniejszego napędu, żeby nie przeskakiwać
źródeł między decyzjami. Plastyczność LIVE wymaga niedawnego pobierania
nikotyny; sama nikotyna pozostająca w organizmie nie uruchamia kolejnego LTD.
PRE/TRAIN/TEST zachowują poprzedni adapter do porównań eksperymentalnych.

## Losowa zielona arena

Każde nowe uruchomienie LIVE losuje pozycje papierosów, drzew, kamieni
i kępek trawy. Losowanie jest niezależne od `--seed` mózgu, więc wcześniejsze
checkpoints nadal działają. `--arena-seed 42` odtwarza ten sam układ,
`--cigarette-count 8` zmienia liczbę losowanych papierosów (1–12),
a ręczne `--cigarette-position` zastępują losowanie papierosów.
Ziarno i pozycje są wypisywane w terminalu. Pełna scena jest zapisywana
przed uruchomieniem fizyki w `A_LIVE_0_arena.json` (także przy przerwaniu).

Papierosy są rozdzielone o co najmniej 8 mm; ich sąsiedztwa zapachu
tworzą połączony układ ze startem. To nie gwarantuje znalezienia wszystkich
źródeł przez aktualny model. Dekoracje domyślnie nie mają kolizji. `--solid-scenery` eksperymentalnie
włącza kolizje pni i kamieni; korony i trawa pozostają dekoracją. Między źródłami oraz przy starcie pozostają szerokie
przejścia, ponieważ model nie ma osobnego sterownika omijania przeszkód.
`--no-scenery` przywraca płaską arenę bez dekoracji.

```bash
MUJOCO_GL=egl .venv/bin/python experiments/smoking/smoking_experiment.py \
  --live --preview --conditions A --duration 30 --arena-seed 42 \
  --weights-dir results/smoking/run_20261007_161611_485707 \
  --out results/live
```

Pomiń `--arena-seed`, żeby każde uruchomienie miało inny układ.

## Szukanie od startu i granica areny

Wcześniejsze sterowanie przeznaczone do pierwszego stałego źródła nie jest
używane w losowej arenie LIVE. Wyuczona reakcja na sam zapach jest mierzona
przed startem i stosowana od pierwszej decyzji; pomiar nie zmienia wag ani
stanu sieci. Bez wyuczonej reakcji nie jest wymuszane przyciąganie.
Dekoracje domyślnie są bez kolizji, bo model nie ma omijania przeszkód.
Przy zewnętrznej granicy obszaru włącza się odruch obrotu w stronę wnętrza.
To osobny adapter granicy, nie kierowanie do papierosa. Stan `BOUNDARY TURN`
i kolumna `boundary_recovery` pozwalają odróżnić go od szukania zapachu.
Podłoże renderowane jest szersze niż obszar ruchu; granice zapisujemy
w `*_arena.json`.

## Wychodzenie z utknięcia

LIVE obserwuje zmianę pozycji podczas aktywnego marszu. Jeśli przez 1,2 s
mucha nie oddali się o 0,5 mm od punktu początku okna, uruchamia
0,45 s cofania, 0,6 s obrotu i 0,45 s mocniejszego kroku. To rzeczywiste
polecenia do kontrolera nóg, bez teleportowania i zmian pozycji w fizyce.
Kolejne próby zmieniają stronę obrotu. Palenie i obrót graniczny anulują
wykrywanie oraz manewr; wejście w strefę karmienia przerywa go także
wewnątrz kroku decyzyjnego. Po manewrze zaczyna się nowe okno obserwacji.
Na obrazie pojawia się `UNSTUCK - REVERSE/TURN/RESUME`. CSV zapisuje
`recovery_state`, `recovery_attempts` i końcowe sygnały napędu nóg
`descending_left/right`. Odruch nie gwarantuje wyjścia z każdej awarii
fizyki i nie zmienia wyuczonej atrakcyjności zapachu.

## Podnoszenie, ogień i zapalanie (domyślny LIVE)

Aktualny LIVE wykonuje `SEARCH_CIGARETTE → SEARCH_FIRE → IGNITING → SMOKING`.
Przy kontakcie automat podnosi jeden papieros i usuwa go z pola zapachu
na ziemi. Mucha niesie widoczny biały papieros z filtrem przed głową. Jest
to kinematyczne przyczepienie obiektu, nie symulacja chwytu nogami.
Sterownik wybiera najbliższy ogień i kieruje do niego muchę. Po 0,6 s
kontaktu z ogniem zapala papieros: pojawiają się żar i dym. Dopiero wtedy
zużywa zapas i pobiera nikotynę. Po zużyciu pozostaje niedopałek i mucha
wraca do szukania. Zapach niesionego papierosa pobudza te same ORN, dzięki
czemu sygnał PAM podczas palenia może nadal wzmacniać skojarzenie zapachowe.

Sekwencja i wybór ognia są zaprogramowanym automatem zachowania, a nie
wyuczoną umiejętnością konektomu ani syntetyczną siecią neuronową.
Plastyczność KC→MBON i wpływ nikotyny na PAM zachowują dotychczasowy model.
PRE/TRAIN/TEST nie są zmienione; ich wyniki nie testują nowej sekwencji.

Domyślnie są dwa losowane źródła ognia. `--fire-count 3` zmienia liczbę
(1–8), a powtarzane `--fire-position X Y` zastępują losowanie. F1, F2 itd.
są oznaczone na minimapie; pozycje są zapisane w `*_arena.json`.
CSV zapisuje `smoking_cycle_state`, `carried_cigarette`, `finished_cigarettes`
i `ignition_progress_sec`. Zapas nie maleje podczas noszenia i zapalania.
`--legacy-contact-smoking` przywraca stary tryb pobierania przy kontakcie
bez ognia. Starsze opisy tego mechanizmu w README dotyczą tego trybu.

```bash
MUJOCO_GL=egl .venv/bin/python experiments/smoking/smoking_experiment.py \
  --live --preview --conditions A --duration 60 --fire-count 3 \
  --weights-dir results/smoking/run_20261007_161611_485707 \
  --out results/live
```

## Stabilny chód LIVE

Napęd podczas marszu LIVE wynosi 0,8 zamiast 0,3. Zmiana polecenia
nóg jest ograniczona do 4 jednostek na sekundę fizyki. Odzyskiwanie ruchu
i zawracanie przy granicy używają dodatnich sygnałów o różnych amplitudach,
bez nagłego odwracania częstotliwości nóg. Aktualny manewr ma etapy
`REPOSITION → TURN → RESUME`; zastępuje opisane wcześniej cofanie.
Po błędzie zachowujemy ukończone decyzje CSV, wagi, informację o błędzie
i dostępne klatki w plikach `*_partial.*`. Błąd nadal przerywa przebieg,
żeby uszkodzona fizyka nie trafiała do wyników jako poprawna symulacja.

Po zużyciu papierosa automat przechodzi teraz do `LEAVE_FIRE`: oddala się
co najmniej 4 mm od wszystkich źródeł ognia przed powrotem do szukania.
Ten etap jest zaprogramowanym odejściem od płomienia, nie uczeniem mózgu.
Napęd i odzyskiwanie ruchu są aktywne; nie pobiera nikotyny i nie podnosi
następnego papierosa, dopóki nie wyjdzie z okolicy ognia.

## Długie przebiegi i dekoracje bez kontaktów

Domyślny LIVE trwa teraz 300 s fizyki (5 minut); przy `--play-speed 0.5`
film ma około 10 minut. Jawne `--duration 30` nadal skraca go do 30 s;
użyj `--duration 300` lub pomiń tę opcję. Czas obliczeń jest znacznie dłuższy
niż czas fizyki. Nagranie jest kodowane na bieżąco do MP4, bez przechowywania
całego filmu w RAM. Monitor mózgu przechowuje liczniki spike’ów, bez historii
czasów wszystkich zdarzeń. Minimapę ograniczamy do 2000 ostatnich punktów.
Przerwanie Ctrl+C zachowuje częściowy film, CSV i wagi.

FlyGym automatycznie tworzył jawne pary kontaktów nóg z geometrią areny,
co omijało `contype=0` dekoracji. `ExperimentFly` usuwa te pary przed
kompilacją fizyki: trawa, korony, papierosy, dym i płomienie nie mogą
blokować nóg. Pozostają kontakty z podłożem. Opcja `--solid-scenery`
dopuszcza dodatkowo kontakty z pniami i kamieniami.

## Szukanie kolejnego papierosa za muchą

Adapter LIVE używa teraz czterech lokalnych próbek pola zapachu: przód,
tył, lewo i prawo. Wyuczona reakcja MBON nadal wybiera przyciąganie
lub unikanie; kierunek wynika z różnic pola. Próbka tylna umożliwia
rozpoznanie źródła za muchą, a normalizacja kierunku nie tłumi skrętu
przy słabym zapachu. Nie są przekazywane pozycje ani numery celów.
To jawny sensoryczny adapter ruchu, nie nowy obwód neuronów konektomu.
Po odejściu od ognia może zawrócić do pozostałych niewyczerpanych źródeł.
Nowe pliki areny zapisują `contact_policy` i `navigation_adapter`.
Dodatkowa kontrola skompilowanych kontaktów przerywa start, jeśli
dekoracja zostanie błędnie włączona jako powierzchnia dla nóg.

## Film w podglądzie VS Code

Zapis strumieniowy używa teraz H.264 (`libx264`) w MP4, formatu pikseli
`yuv420p` i nagłówka `faststart`. Poprzedni kodek `mp4v` tworzył pliki,
które dało się dekodować np. w OpenCV, ale nie w podglądzie VS Code.
Plik w trakcie zapisu ma końcówkę `.recording.mp4`; końcowe `A_LIVE_0.mp4`
pojawia się dopiero po zamknięciu i sfinalizowaniu kodowania. Podczas
obliczeń korzystaj z `--preview`. Po przerwaniu kodowania przez Ctrl+C
film jest zamykany i przenoszony do `*_partial.mp4`.
