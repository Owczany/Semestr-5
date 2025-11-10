import math
import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "flax-community/papuGaPT2"
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()


print("MODEL LOADED")

@torch.inference_mode()
def continuation_logprob(prefix_text: str, continuation_text: str) -> float:
    """
    Zwraca log P(continuation | prefix) w jednostkach 'nats',
    licząc sumę po tokenach kontynuacji (autoregresywnie).
    """
    # Tokenizujemy prefix i kontynuację osobno, potem łączymy
    prefix_ids = tokenizer(prefix_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(DEVICE)
    cont_ids = tokenizer(continuation_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(DEVICE)

    # Sklejamy wejście: [prefix][continuation]
    input_ids = torch.cat([prefix_ids, cont_ids], dim=1)
    # Forward
    out = model(input_ids=input_ids)
    logits = out.logits  # shape: [1, T, V]

    # Interesują nas logity dokładnie w tych pozycjach, gdzie model przewiduje tokeny kontynuacji
    # Dla kontynuacji o długości L patrzymy na ostatnie L pozycji logitów poprzedzających każdy token kontynuacji
    L = cont_ids.size(1)
    # Indeks startu kontynuacji w input_ids
    start = input_ids.size(1) - L

    # Bierzemy logity dla pozycji [start-1 .. start+L-2], czyli przed każdym tokenem kontynuacji
    # Technicznie: logits[:, pos-1, :] -> przewidywanie dla tokenu na pozycji 'pos'
    # Zbudujemy iteracyjnie, żeby nie gubić zależności autoregresywnej.
    logp_sum = 0.0
    # Bieżące wejście, które będziemy rozszerzać token po tokenie (symulacja teacher forcing)
    cur_input = prefix_ids.clone()

    for i in range(L):
        out_i = model(input_ids=cur_input)
        logp = F.log_softmax(out_i.logits[:, -1, :], dim=-1)  # predykcja następnego tokenu
        token_i = cont_ids[:, i]  # oczekiwany kolejny token kontynuacji
        logp_i = logp.gather(1, token_i.view(1, -1)).squeeze().item()
        logp_sum += logp_i
        # dołóż prawdziwy token, żeby przewidywać następny
        cur_input = torch.cat([cur_input, token_i.view(1, -1)], dim=1)

    return float(logp_sum)

def tmpl_1(opinia: str):
    prefix = opinia.strip()
    if not prefix.endswith("."):
        prefix += "."
    prefix += " Ta opinia jest "
    return prefix, "pozytywna.", "negatywna."

def tmpl_2(opinia: str):
    prefix = f"Opinia: {opinia.strip()}"
    if not prefix.endswith("."):
        prefix += "."
    prefix += " Werdykt: "
    return prefix, "pozytywna.", "negatywna."

def tmpl_3(opinia: str):
    prefix = opinia.strip()
    if not prefix.endswith("."):
        prefix += "."
    prefix += " Ogólnie "
    return prefix, "polecam.", "nie polecam."

def tmpl_4(opinia: str):
    prefix = opinia.strip()
    if not prefix.endswith("."):
        prefix += "."
    prefix += " To doświadczenie było "
    return prefix, "dobre.", "złe."

def tmpl_5(opinia: str):
    prefix = opinia.strip()
    if not prefix.endswith("."):
        prefix += "."
    prefix += " Ocena jest "
    return prefix, "dobra.", "złaa."

TEMPLATES = [tmpl_1, tmpl_2, tmpl_3, tmpl_4, tmpl_5]

def score_opinion(opinia: str) -> dict:
    template_scores = []
    logp_pos_total = 0.0
    logp_neg_total = 0.0

    for t in TEMPLATES:
        prefix, pos_cont, neg_cont = t(opinia)
        lp_pos = continuation_logprob(prefix, pos_cont)
        lp_neg = continuation_logprob(prefix, neg_cont)
        template_scores.append((t.__name__, lp_pos, lp_neg, lp_pos - lp_neg))
        logp_pos_total += lp_pos
        logp_neg_total += lp_neg

    return {
        "logp_pos": logp_pos_total,
        "logp_neg": logp_neg_total,
        "margin": logp_pos_total - logp_neg_total,
        "template_scores": template_scores,
    }

def predict(opinia: str) -> str:
    sc = score_opinion(opinia)
    return "pos" if sc["logp_pos"] > sc["logp_neg"] else "neg" #+ f" Oceny szablonów: {sc["template_scores"]}"

positives = [
    'Lokalizacja w centrum - przy głównym deptaku, po sąsiedzku Biedronka : ) MINUSYDługie schody do pokonania ( od wejścia głównego hotelu do recepcji ).',
    'Hotel jest położony prawie nad samym jeziorem, u jego początku.',
    'Warto przejechać ponad 600 km żeby przez tydzień wypocząć w tym hotelu : )',
    'Wypożyczalnia samochodów w firmie hotelowej, również godna polecenia.',
    'Butelkę mamy 3 miesiące i nic się ze smoczkiem nie stało.',
    'Do dziś są wdzięczne mnie za polecenie doktora, a doktorowi za to są szczęśliwymi matkami.',
    'Pozytywni ludzie i pozytywne wspomnienia pozostaną w naszej pamięci na zawsze.',
    'Wiem to doskonale na swoim przykładzie, bo mój uśmiech został oszpecony przez innych lekarzy i dopiero dr Fudalej doprowadził go do obecnego stanu.',
    'Przedstawił mi wszystkie możliwe opcje w związku z moim leczeniem.',
    'Obsługa hotelu bardzo przyjaźnie nastawiona, miła, uprzejma i pomocna, zawsze z uśmiechem.',
    'Nie tworzy efektu maski, świetnie stapia się ze skórą.',
    'Droga do szczęścia jest trudna, ale dzięki takim ludziom, łatwiej ją pokonać.',
    'Są domki, pokoje i apartamenty.',
    'Serdeczne pozdrowienia dla Pani Dr Dominiki Tuchendler oraz Dr Szymczaka, który zaraża swoim spokojem i opanowaniem pacjentów.',
    'Za 600 zł para, to jest adekwatna cena co do ich możliwości.',
    'Dobra lokalizacja w Centrum miasta,',
    'Poleciłam już ten gabinet kilku naszym znajomym i jakoś wszyscy są zadowoleni.',
    'Lekarz jest skrupulatny, precyzyjny i bardzo zdecydowany w swoich zabiegach.',
    'Polecam ten hotel - świetny widok, super dojazd.',
    'Trzydniowy pobyt w Pensjonacie pod Kotelnicą był bardzo udany!',
    'I muszę powiedzieć, że nie żałuję.',
    'Na terenie basenu sa ogolnodostepne akcesoria do plywania przydatne szczegolne dla dzieci ( deski, dzieciece kolo ratunkowe, zarekawki ).',
    'Dodam, że mimo silnego trądziku, nie przepisano mi na początek leku ostatecznego - izoteku - najpierw były to inne środki.',
    'Kurs : Analiza Matematyczna I i II, wyklad nr 1 [ / b ] Oto najciekwszy prowadzacy z jakim mialem stycznosc na Pwr.',
    'Mini disco dla dzieci pierwsza klasa!',
    'Bardzo sympatyczny kucharz.',
    'Do plaży 5 minut spacerkiem, uroczą uliczką wśród palm.',
    'Na szczęście trafiłam do Pana Doktora, zostałam przyjęta o czasie ( jeszcze nigdy, nigdzie mi się to nie zdarzyło ), Pan Doktor przepisał mi leki które mnie wyleczyły w dwa dni (! ).',
    'W obiekcie jest mały basen z przeciwprądem i dyszami do masażu.',
    'Ten lekarz czyni cuda! ~ 10lat nie bylam u dentysty bo sie panicznie balam, a Pani doktor sprawila ze leczenie zebow jest przyjemne.',
    'Polecamy wycieczki z przewodniczką Bożenką jest wspaniale.',
    'Pani Doktor jest bardzo uważna i wnikliwa.',
    'Pani Dotkor bardzo uprzejma, miła i sympatyczna.',
    'Obiady i kolacje codziennie inne.',
    'Osrodek ma wiele atrakcji dla dzieci a ilosci dzieciakow sprawia ze dzieci moga " zniknac " na kilka godzin ; ) Najważniejsze to super Mila rodzinna atmosfera.',
    'Działał mi bardzo ładnie, nie krzaczył się jak na moje wymagania - igła.',
    'To człowiek bardzo wysokiej klasy.',
    'Pokoje są przestronne, łóżko wygodne, rozłożyste z przyjemną w dotyku pościelą.',
    'Proszę się nie zrażać pierwszym wrażeniem i pójść na kolejna wizytę, a będziecie zadowoleni.',
    'Jestem przekonana, że jeszcze tam wrócimy, na dłużej.',
    'Doktor Ufnalewski prowadzi moją już drugą zagrożoną ciąże.',
    'Pani Doktor zleciła szereg badań, nie koncentrując się tylko na tych związanych z tarczycą ( leczyłam się u innego lekarza na niedoczynność, a mimo to byłam wiecznie zmęczona, ospała ), wzięła pod uwagę również inne możliwości, m. in. niedobór witaminy D w organizmie.',
    'Bardzo, bardzo miły hostel i przede wszystkim wspaniali ludzie, młodzi, życzliwi i tacy gościnni!!!',
    'Polecam ją z całego serca!',
    'Jako jedyny lekarz podszedł do mnie poważnie.',
    'Co dziennie zmieniane ręczniki, pościel co dwa dni.',
    'B. Szybko ładują się programy i to mi wystarcza póki co.',
    'Pracownicy recepcji bardzo przyjaźnie nastawieni i chętni do udzielania wszelkich informacji.',
    'Używam go do panasonica lumix lx7, aparat niestety 3 razy mi spadł na szczęście zawsze zapakowany był w dany pokrowiec, ktory idealnie go osłonił.',
    'Bardzo przyjazny personel, szczególne podziękowania dla pana Giuseppe który wyjaśnił sprawę transferu.',
    'Pokoje te są zlokalizowane na piętrze pensjonatu, co jest zaletą.',
    'Przez całe dwa lata leczenie przebiegało prawidłowo, wizyty zawsze na czas, nigdy nie było opóźnienia ani przesunięcia terminu.',
    'Miłym zaskoczeniem była spora ilośc drzew i brak wysokich płotów - nareszcie ktoś pomyślał, że płoty i brak zieleni to coś co mamy na codzień...',
    'hotel bardzo przyjemny, położony w dogodnym miejscu, bardzo fajny ogródek z wyjściem na deptak położony na tyłach hotelu, pokoje ładne, bardzo czyste, medialne.',
    'Posiłki są super!',
    'Położenie super, wyjście z hotelu bezpośrednio na promenadę nad Wisłą.',
    'Do tego szybki i darmowy internet ( wifi ).',
    'Ten hotel jest położony w doskonałej lokalizacji.',
    'Pytał o opis usg, o opinię kierującego.',
    'Bardzo się ucieszyliśmy, że będziemy mogli mieć wspólny pokój.',
    'Pokoje przestronne i wygodne.',
    'Pochwalić muszę śniadania i organizację, bo choć bywały rano tłumy to pracownicy naprawdę błyskawicznie wszystko ogarniali.',
    'Smaczne, różnorodne, obfite.',
    'Nawet jeżdżące pod oknem tramwaje nie zepsuły przyjemności nocowania w tym miejscu : ) Zdecydowanie polecam!',
    'Przeprowadza dokładny wywiad, kieruje na wszystkie badania ( z czym inni lekarze często mają problemy ), nie spieszy się na wizytach tylko bardzo dokładnie i wyczerpująco wszystko tłumaczy i odpowiada na wątpliwości.',
    'Jedzenie doskonale.',
    'Ładnie położony na wzgórzu, częściowy ( przed hotelem jest inny hotel ) widok na morze i Złote Piaski ( trochę schodów do plaży w dół i w górę, ale to dobre dla zdrowia w opcji ALL ), ładne i czyste pokoje.',
    'Pan Grzegorz odpowiada na moje maile i wyjaśnia mi wątpliwości.',
    'Polecam sprzęt od MAKITA.',
    'Nawiązując do posiłków, w tawernie XL z pięknym widokiem na morze - jedzonko super : sałatka grecka, musaka, sola i inne rzeczy, wspaniałe, świeżutkie i przesmaczne.',
    'A plusy : 1.',
    'Wielokrotnie pomagał moim dzieciom Oldze i Justynie Pawlak, zawsze był żywo zainteresowny ich zdrowotnymi problemami, co jest obecnie rzadkością.',
    'Po rocznej terapii udało nam się nadrobić wszystkie zaległości, corka z chęcią chodziła na zajęcia ale najważniejsze było dla mnie to, że z wizyty na wizytę było widać efekty " naszej " pracy.',
    'To człowiek o wielkiej wiedzy i autorytecie, ale tylko człowiek.',
    'Sniadanie bardzo smaczne i obfite.',
    'Pokoje pięknie wyglądają na zdjęciach.',
    'Zdecydowanie polecam ten hotel.',
    'Pani Doktor Iwona jest wyrozumiała i delikatna, bardzo miła, cierpliwie wyjaśnia każdy zaistniały problem a co najbardziej mi się podoba to fakt, że w swoim działaniu jest bardzo stanowcza i konkretna.',
    'Podejście Pani doktor do pacjenta to pełen profesjonalizm.',
    'Dobrze zaopatrzony sklep na terenie hotelu.',
    'Jego największą zaletą jest Aquapark z rozbudowaną strefą saun we wszystkich możliwych wariantach, grotą solną i ze słoneczną łąką, gdzie rzekomo można złapać nieco opalenizny leżąc pod lampami UV.',
    'Przy hotelu zamek i zjeżdżalnie dmuchańce w cenie pobytu.',
    'Bardzo duże i czyste apartamenty, zaaranżowane z prostotą i ze smakiem.',
    'Diabelnie szybki, perfekcyjnie wykonany, pięknie wykończony, bardzo funkcjonalny.',
    'Plus za to, że przyjmuje w gabinecie prywatnym zawsze, nie ważne czy święto czy nie, w nagłych przypadkach to jest bardzo wygodne.',
    'Obsługa za to bardzo sympatyczna : )',
    'Można natomiast dostać wrzątek w recepcji hotelu.',
    'Największą zaletą jest niewątpliwie kuchnia.',
    'Syn urodził się zdrowy za co jesteśmy dozgonnie wdzięczni!',
    'Blisko do centrum.',
    'Jedzenie w porządku.',
    'jestem zadowolona z wizyt u Pani Ewy, jest to już moja kolejna wizyta, jest osobą miłą, kuracja dot. mojego problemu przynosi efekty, wcześniej leczyłam się u pięciu innych lekarzy, ale problem po zaleczeniu szybko się powtarzał, na razie minął rok po pierwszej wizycie i nie zapeszając : ) jest lepiej...',
    'Co ważne, butelka posiada korek.',
    'Otoczenie hotelu zadbane ( przystrzyżone trawniki i żywopłoty, myte codziennie leżaki wokół basenu ).',
    'Zadzwonilam na recepcje mila pani recepcjonistka powiedziala ze zaraz doniesie reczniki, tak tez zrobila.',
    'Ogólnie polecam : ]',
    'Dla mnie najważniejsza jest ekologiczna misa z powłoką ceramiczną, bo jestem mocno wyczulona na zapachy przenikające do jedzenia.',
    'Wprowadza atmosferę spokoju i rzeczowości.',
    'Bardzo polecam dla rodzin z dziećmi.... . duży plac zabaw blisko Park atrakcji, do miasta miły spacerek przez lasek... obsługa bardzo miła i pomocna Pokoje czyste, śniadania i kolacje na wysokim poziomie każdy znajdzie coś dla siebie Bardzo polecam strefę SPA, na basenie dodatkowo są ręczniki co nie zawsze się spotyka. .',
    'Śniadanie bardzo poprawne - duży wybór, miła obsługa - nie ma się do czego przyczepić.',
    'Spóźniliśmy się na śniadanie, miła pani z recepcji ostatecznie nas wpuściła na śniadanie : )',
    'Ładne kolorki, możliwość rozkładania siedziska to również atuty.',
    'Jestem pewny, że to najlepszy dermatolog w Warszawie.',
    'Pokoje świeżo po renowacji, dostępne dodatkowe poduszki, koc, łazienka czysta.',
    'Wizyty przebiegały w atmosferze bezpieczeństwa.',
    'Ocene podwyższa za referat lub wystąpienie w przypadku seminarki.',
    'Jedyny lekarz, który faktycznie szukał rozwiązań.',
    'Zaleta - duży telewizor i zestaw do kawy i herbaty.',
    'Okolice hotelu, plaża Antoniego Quina, Faliraki, miasto Lindos za zatoką świętego Pawła oraz zamek Monolitów są cudowne.',
    'Ogólnie - polecam',
    'Najlepszy Sylwester na jakim byłam.',
    'Piekne miejsce, delikatne dekoracje slubne dodatkowo podkreslily urok sali.',
    'Jestem 2 dni po zabiegu i naprawde widac poprawe owalu twarzy.',
    'Bardzo luzacki prowadzący.',
    'Nocowaliśmy w starszej części hotelu - spore, czyste pokoje.',
    'Co prawda miałam za mało zacięcia, żeby dojść do końca, ale pierwsze lekcje, które przerobiłam, rzeczywiście dużo mi dały.',
    'Polecam wszystkim wyjazd do tego sanatorium, a nie za granice, gdzie jest drozej i nie wiadomo, czy sie wroci do domu, bo biura podrozy upadaja.',
    'Polecam i mam nadzieję, że taki poziom już pozostanie.',
    'Ale dość dobre więc nie ma co narzekać, bo dla każdego się coś znajdzie.',
    'Ale wygląda naprawdę świetnie : )',
    'Mają także kilka rowerów do wypożyczenia także nie trzeba nigdzie daleko szukać.',
    'Króliki biegaja po terenie, przepiękny, kaskadowy ogród z wodospadami.',
    'Generalnie polecam ten hotel, wróciliśmy wypoczęci i zadowoleni!',
    'Hotel bardzo ładny.',
    'No i będzie dobrze.',
    'Pan Robert umożliwił mi to.',
    'Na zakończenie leczenia rzeczowa, przyjacielska rozmowa z dokładnymi wskazówkami co dalej czynić, jak postępować w przypadku jakichkolwiek problemów.',
    'Do tego bardzo blisko do Parku Zdrojowego i co najwazniejsze - panuje tu znakomita obsługa.',
    'Nigdy nie zwątpiłam w kompetencje i profesjonalizm Pani Profesor.',
    'Wykupiliśmy pakiet sylwestrowy ze znajomymi w hotelu Lenart i muszę powiedzieć, że była to b. dobra decyzja.',
    'Bardzo uprzejma i młoda obsługa.',
    'Szczerze polecam Marcina.',
    'ale mimo wszystko polecam tą grę bo według mnie powinien ją posiadać każdy posiadacz konsoli PlayStation 3.',
    'A moc przy blendowaniu zmrożonych produktów i twardych rzeczy jest naprawdę duużym plusem tego robotu.',
    'Oczywiście nic nie dzieje się bez udziału pacjenta i trzeba samemu się zaangażować w terapię, ale dużą zaletą Pani Moniki jest to że tak umiejętnie potrafi zadać odpowiednie pytania że skłania do refleksji, pomaga spojrzeć na własne problemy i przyjrzeć się swoim odczuciom z różnych stron, co mi bardzo pomogło.',
    'Nie wiem, możę pani ma jakies inne wymagania ale jak dla mnie jest super.',
    'Niezwykłe miejsce, hotel usytuowany na górze, wykuty w skale, prowadzi kręta droga wjazdowa, hotel wpisany w skały, z przepięknym widokiem na miasto w dole i na góry, skały!!',
    'Ja daję 5 gwiazdek, ponieważ moim zdaniem to świetny kosmetyk.',
    'Spokojnie wszystko tłumaczył, zwłaszcza, że pacjentką była 86 letnia osoba.',
    'Widać, że Pani Doktor dokładnie analizuje każdy przypadek i wybiera najlepsze rozwiązania.',
    'Bajkoterapia działa!',
    "wymiana takiej sprężynki to koszt 1zł i można to zrobić w każdym sklepie victorinox'a.",
    'Obsługa hotelu bardzo miła i pomocna, organizacja imprezy bardzo dobra.',
    'Posiada ogromną wiedzę i doświadczenie.',
    'Dla nas to akurat była atut.',
    'Bardzo ciche miejsce, znajdujące się na uboczu.',
    'Ręczniki i pościel zmieniane codziennie, jedzenie bardzo smaczne i bardzo dużo.',
    'Byłam pod ogromnym wrażeniem jej pamięci.',
    'Pierwszy raz spotkałam się z tym, że w każdej turystycznej miejscowości są informacje dla turystów w j. polskim.',
    'Funkcjonalność zestawu jako bezprzewodowego do rozmów jest zgodna z opisem technicznym, zajmę się więc jego stroną fizyczną.',
    'Chciałem wrócić do tego świata, tych bohaterów i emocji.',
    'Raptem kilkaset metrów od dworca a także stacji metra Florenc.',
    'Śniadania na duży plus od godziny 8 00.',
    'Tak Podstawa oceny w indeksie : egzamin Niespodziewajki : brak ( zadania powtarzają się co roku ) Poczucie humorupane pokłady Ogólne podejście do studenta : pozytywne Zerowy termin jest najprostszy ( 3 zadania ) - warunkiem przystąpienia jest jednak zaliczenie projektu u dr.',
    'Uspokoił nas.',
    'Szczerze polecam hotel dla osób które jadą do Rzymu z myślą o zwiedzaniu',
    'Obsługa miła i życzliwa.',
    'Zobaczenie znajomej twarzy w sali pełnej obcych ludzi w dość nieprzyjemnej sytuacji ( nagła diagnoza do natychmiastowego rozwiązania ) - cesarkę zrobił przepięknie, prawie nie mam śladu po cięciu.',
    'Najlepsze bylo na koncu.',
    'Do dziś używam kiedy tylko poczuję, że się coś dzieje.',
    'Natomiast obiady fantastyczne, pyszne i porcje zadowalające mężczyzn.',
    'polecam, chodzę do Niej już 10 lat - jestem bardzo zadowolona, prowadziła mi kilka lat temu ciążę - zlecała wszystkie potrzebne badania, USG - idealnie " wymierzyła " termin porodu.',
    'Nowoczesne, czyste, zadbane zamknięte osiedle.',
    'Pianka też.',
    'A jeśli chodzi o sam zabieg i moje samopoczucie po - również mogę o nim pisać w samych superlatywach.',
    'Dobrze szłyszalny wokal.',
    'Bardzo dobry lekarz, delikatny, miły i kompetentny.',
    'Wstęp do Algebry i Geometrii Ciocia Dorotka - bo taki pseudonim dostała od naszej grupy, to chyba najlepszy prowadzący z C - 11.',
    'Ogólnie BwE to pikuś 8 )',
    'A warto ten czas znaleźć.',
    'Śniadania obfite i urozmaicone.',
    'U nas raz puścił liste ale potem stwierdził ze dalej będzie czytał bo tak szybciej ( niedziwie sie ludzie sobie wpisywali obecności wcześniej a nawet na zapas : D hehe ) Miałem 3 nieobecności na 10 chyba i dał mi 4.5 Można sie z nim dogadać lu zwolnienie mu przynieść : D UZNAJE : D',
    'Spring, Reverse i dwa Pitch Shiftery w pełni ustawialne - fajnie.',
    'Dzięki Panu Włodarczykowi już nie odczuwam za każdym razem lęków : Mogę śmiało zjeżdżać ze ślizgawki w " aquaswerze ", nawet z rekinem ludojadem pod pachą.',
    'Ten niepozorny hotel potrafi mile zaskoczyć.',
    'Jest przy tym uważna, życzliwa, troskliwa, pełna wiary w wyleczenie i ta wiara udziela się pacjentowi.',
    'Zupełnie inne podejście miała Pani doktor na nocnym dyżurze w przychodni – dokładniej obejrzała dziecko, osłuchała i „ wymacała ” brzuszek w celu wykluczenia innych przypadłości co w przypadku małego dziecka jest bardzo ważne.',
    'Gdy się chce czegoś u niej nauczyć - nie ma problemu ( ale trzeba faktycznie samemu wykazać chęci, bo nie ma do niczego przymusu ).',
    'Wszystko pracuje gładko i płynnie.',
    'Jest to roczny ( oddany do użytku 06 / 2017r. ) pensjonat więc wszystko jest nowe, czyste, zadbane.',
    'Mimo długiego weekendu i zawodów hippicznych ścisku nie ma.',
    'Dobrą stroną jest wyodrębniony brodzik dla mniejszych dzieci i sporo sprzętu wodnego.',
    'Jesli mielibysmy wrocic do Zakynthos to tylko i wylacznie do Alykanas do naszych przyjaciol prowadzacych Studio MOUZAIKIS i tawerne LEVANTE....',
    'Czysto jak na 2 - gwiazdkowy hotel.',
    'Rewelacyjnie puder ten sprawdza się latem.',
    'Polecam gorąco!',
    'Jak na trzy gwiazdki po prostu świetnie.',
    'Miejsca jest dla wszystkich.',
    'Codziennie ryby lub owoce morza, mięso i potawy wegetraiańskie, kącik dla dzieci.',
    'Herbata, kawa i siłownia zawsze pod ręką, co bardzo umila nawet dłuższy pobyt.',
    'Pomijając obsługa sali bardzo miła i pomocna.',
    'Jej delikatność, ciepło, jak i pełne zdecydowania podejście do problemu sprawiają, że nasze sesje terapeutyczne stają się dla mnie bezpieczną podróżą do wewnętrznego świata moich najgłębszych przeżyć, potrzeb i pragnień, o istnieniu których nie miałam do końca pojęcia.',
    'Wszystkim, którzy mają schorzenia urologiczne polecam Pana dr Bartosza Małkiewicza.',
    'Dobra komunikacja z miastami.',
    'Obsługa bardzo miła, pokoje ładne i czyste.',
    'Fantastyczna lokalizacja budynku a także ciekawe elementy wystroju z czasu - naszego - socjalizmu.',
    'Pokoje obszerne, czyste, wyposażone w czajnik z zestawem herbat.',
    'mieszkając właśniena na ul. ŻELAZNEJ wraz ze schorowaną leżącą mamą mająca 87 lat Pan dr. kamyk otoczył swoją opieką medyczną moją mamę i chcę powiedzieć jedno nie spotkałam w życiu tak oddanego lekarza wykoującego swój zawód w stosunku do osób starszych jego wielkie serce,. zaangazowanie w leczeniu mojej mamy.',
    'Mocna bateria jak na smart.',
    'Kilka metrów od brzegu znajduje się pływająca platforma na której można się poopalać.',
]

negatives = [
    'Potrafi przyczepic sie do wszystkie - osie symetri za grubym olowkiem rysowane, wkrecajac srube w drewniany otwor z gwintem nalezy narysowac wybrzuszenie drewna po drugiej stronie ( w ksiazce nic o tym nie ma ) i wlasnie na tej podstawie cie obleje...',
    'Trochę mnie to zdziwiło, ponieważ z tego co mi wiadomo to są to moje zęby a nie lekarza, więc z łaski swojej mogła Pani doktor chociaż zapytać czy mam ochotę zrobić dwa zęby jednoczesnie i co też istotne czy jestem przygotowana finansowo na to.',
    'Apartamenty znajdują się w budynkach, które położne są w zupełnie innym miejscu niż hotel!!!',
    'Zaskakujące jest, że takie miejsce w ogóle istnieje.',
    'Po 20 minutowej miłej rozmowie w gabinecie, wypisaniu recepty manualnym zbadaniu wątroby i zapłaceniu 150 złotych jestem lekko zawiedziony aby nie napisać rozczarowany Pozdrawiam Rafał. K',
    'Konfiguracja z obsługą WiFi pada przy próbie instalacji skanera.',
    'Uwaga nie w wakacje.',
    'Na nic nie zdają się, żadne smary czy oleje specjalistyczne do tłoków itp.',
    'Mało bezpiecznie, ale przynajmniej nie zamarzliśmy.',
    'Znowu zostałam w punkcie wyjścia i nie mam już ochoty opowiadać kolejnemu psychologowi swojej historii.',
    'Jedynym minusem jest brak akceptacji kart Diners oraz gotówki innej niż lokalna.',
    'Ja osobiście z takimi maluchami nie wybrał bym się tam... bo trzeba naprawdę sporo się nachodzić.',
    'Łazienki na terenie SPA wygladaja dosyć nieświeżo.',
    'Jeśli nie masz dzieci wykończysz się tutaj zamiast wypocząć po prostu nie przyjeżdżaj tu.',
    'Nie dość, ze na termin wizyty długo sie czeka ( ja miesiąc ) to ten spęd jaki tam jest w środe po poludniu ( lekarz przyjmuje TYLKO 2h i spoznil sie 30min... )',
    'Niestety ale trąbi tu chór rozanielonych użytkowników a prawda jest taka że zawiodłem się okrutnie.',
    'Już na pierwszej wizycie dostałam leki antydepresyjne oraz miliony maści ( w tym maść przeciwbólową ) - doktor stwierdził wulwodynię, chociaż nie byłam pewna czy słuchał tego, że jej objawy nie pokrywają się z moimi.',
    'Lubię chwilę po obudzeniu podrzemać i posłuchać radia, przy tym się nie da dostajesz patelnią w łeb i lecisz wyłączyć to paskudztwo.',
    'Zrezygnowałam z wizyt w klinice Polis i szukając pomocy udałam się do Lublina, gdzie zdiagnozowano u mnie wulwodynię.',
    'Na razie w sumie konkretnych badań nie robiłam bo jeszcze nie staram się o dziecko, ale jak już się zdecyduje to muszę chyba zmienić lekarza.',
    'Dowiedziałem się, że jeśli szczelina będzie mi przeszkadzać mam przyjść ponownie.',
    "Zamek w drzwiach taki ' niepewny ', futryna po przejściach, jakby ktoś kiedyś ' wjechał z buta ' do pokoju ( przez to nie czułam się tam w pełni bezpiecznie ).",
    'Pokoje sprzątane co 2 - 3 dni!',
    'Szczerze nie polecam!!!',
    'Myślę jednak że w tym roku byłem 1 raz i ostatni.',
    'Na plaży notorycznie wałęsały się psy i człowiek musiał uważać czy jakiś go nie ugryzie.',
    'Na miejscu doznalismy pierwszego szoku by dokonać opłaty za parking w kwocie 5 euro za dobę... gdzie w tui ( tam wykupowalismy hotel ) informowano nas, że parking jest bezpłatny.',
    'Szczerze odradzam, wlascicielka ma sie za wielka dame z milionamii a prawda jest taka ze hotel jest zbudowany z dotacji unijnych.',
    'Sauna sucha - tak designerska, ze nie ma gdzie usiąść.',
    'I do tego zmiana cennika w trakcie trwanie umowy!!!!',
    'cała obsługa " skacze " wokół gości którzy nie mają wykupionej opcji all a zdecydują sie zjeść w hotelowej restauracji ; podobnie było w barze gdzie goscie płacący gotówką dostawali co innego, inaczej podane i w pierwszej kolejności niż " reszta " ; żadnych animacji, niedobór leżaków przy basenach, itp...',
    'Hotel nie powinien mieć ich 5.',
    'Okolica bardzo brudna i zasmiecona, pełna bezdomnych psów.',
    'Przepisanie dawki leku i koniec wizyty.',
    'a tym czasem auto można zaparkować przy ulicy!!!',
    'Niedopracowane same rózgi do misy ( chociaż wyglądają bardzo profesjonalnie ) pozostawiają na dnie nierozmieszane ciasto lub krem ( zaznaczę ze posiadam inne miksery roboty które wypadają o niebo lepiej ) Najbardziej irytujące są jednak mieszadła silikonowe które powinny zbierać wszystko ze ścianek oraz spodu misy co bardzo przeszkadza a właściwie uniemożliwia podbijanie mas na * * * * * z jednoczesnym podgrzewaniem ( w tym celu to zakupiłem ) chyba że chcemy jajecznicę to wtedy są wręcz idealne.',
    'Fatalne łazienki z kamieniem, grzybem, smrodem.',
    'Dwa razy odwołana wizyta, a jak w końcu udało mi się z córką dostać to okazało się ze przyjmuje mnie asystentka, która nie potrafi odnieść się do wcześniejszych zaleceń pani doktor a mimo to kasuje zwyczajnie za wizytę.',
    'Absurdalne było też to, że nie można było często usiąść do konkretnego stolika poniewaz obsluga sobie przygotowała ten stolik na kolejny posiłek, na śniadaniu już sobie przygotowywali na lunch, na lunchu na kolację itd.',
    'Jestem tu drugi raz, i ostatni raz na dworze w nocy ok 4 stopni Celsjusza i nie ma ogrzewania, mimo prośby o jakiś grzejnik elektryczny, to niestety otrzymaliśmy odpowiedź od Pani że mamy koce i taki luksus w tym hotelu za 180 zł',
    'Później było już tylko gorzej …',
    'Nie rozpoznał u mnie astmy.',
    'Kobiety niech wam nawet do głowy nie przyjdzie iść na wizytę to tego człowieka!',
    'Słuchajcie, sprawa wygląda tak : Raczej odradzam zakupu 1 million woda po goleniu, bo zapach bardzo nikły... mimo, iż bardzo bardzo lubię ten zapach, bo mam 1 million wodę perfumowaną ( EDP ), kupioną przez internet ( sklep tylko z oryginalnymi produktami czyli te na ceneo.pl - najlepiej z wysoką ilością osób które kupują produkty ) - odradzam allegro.pl ( w perfumeriach takich sephora lub duglas nie sprzedają wody perfumowanej - dużo lepszej niż woda toaletowa ( EDT ), bo EDP ma dużo trwalszy zapach niż EDT ( czyli woda toaletowa ) - które to EDT sprzedają w tych perfumeriach.',
    'Nie polecam i proponuję trzymać się z dala od tej osoby, jeśli chcecie mieć lekarza zaufanego, który potrafi pomóc w różnych sytuacjach.',
    'Utrudnieniem może być bardzo słaba sień internetowa bezprzewodowa.',
    'Nie zmienia to jednak faktu, że mnie zbył niczym.',
    'W toalecie niedziałająca spłuczka z lejącą się całą noc wodą.',
    'Mam problemy z ładowaniem.',
    'Najgorsze w tym hotelu było że w szczytowym okresie ferii mazowieckim są robione remonty było głośno i pełno robotników poruszających po obiekcie.',
    'POZDRAWIAM PANI DOKTOR ; - /',
    'Tak jak w opinii powyżej zgodzę się, że wykonanie pozostawia wiele do jakości.',
    'Pełno stromych schodów i brak wind, jedzenie niesmaczne, niskiej jakości.',
    'Sorry ale jak na hotel 4 * to slabo.',
    'Wydałam ok 1300zł i na szczęście znalazłam innego lekarza stosujące Dry Needing igłami akupunkturowymi, gdzie po pierwszej wizycie i to tańszej jeszcze w tym samym dniu poszłam na zajęcia sportowe, a następnego dnia na długie zakupy po centrach handlowych.',
    'TEż z ARO.',
    'Poprawcie to.',
    'po godzinie czytania miałam większa wiedzę na temat małwodzia niż dr Łaganowski.',
    'Pani doktor na moją informację o tym, że bezskutecznie staramy się z mężem już rok o dziecko odpowiedziała, że to nie jest poradnia leczenia niepłodności.',
    'Jedzenie okropne, monotonne, typowo przygotowywane pod Brytyjczyków : wszystko smażone, na śniadanie fasola, na obiad i kolację frytki.',
    'Omijajcie szerokim łukiem to dziwne miejsce.',
    'W łazience pojemniki na mydło z płynem na samym dnie, starczyło na jeden prysznic.',
    'Osobiście nie skorzystamy już z oferty tego hotelu.',
    'W trakcie wizyty dowiedziałem się iż ząb jest pęknięty pionowo i nie nadaje się do leczenia oraz iż ząb należy usunąć.',
    'Nie ma możliwości wyłączenia głośnika.',
    'Koty na stołówce i w restauracji!',
    'Doktora Łazęckiego nie polecam.',
    'Przykładowe śniadanie to 2 plasterki sera, 2 plasterki wędliny i 3 plasterki pomidora.',
    'Bede musiał skorzystać z porady innego lekarza, ta wizyta nic nie wniosła, zwraca czasu i pieniędzy.',
    'Inna osoba pisała tu już, że wizyta trwala 3 minuty, a Pan Doktor uznał, że to nieprawda.',
    'Kolejny to problem z talerzami i sztućcami i szklankami na jadalni, jak przyszlo więcej ludzi to stalo się w kolejce czekając na nakrycie.',
    'Impreza u dr Nogala kosztowała mnie 340 zł i wolała bym przeznaczyć te pieniądze na inny cel.',
    'Takie podejście do zdezorientowanego pacjenta uważam za karygodne, bowiem po zleceniu badań od internisty okazało się, że mam ogromny niedobór wit.',
    'Nie polecam innych pokoi - widoki niekoniecznie ucieszą...',
    'Pokoje nie są sprzątane.',
    'Nie polecił bym tego radia, oczywiście wybiera każdy według własnego przekonania.',
    'Już o 22. 50 nic nie można zamówić z baru ( chyba, że za opłatą ).',
    "wizyta trwała 5min, miałam trochę większe oczekiwania wobec Pani Ani, szybkie rozpoznanie ' choroby ' recepta i do domu to za mało. .",
    'Kolejnego dnia chciał bym się wybrać na siłownie lecz niestety był remont i nikt nie raczył poinformować gości.',
    'Pan doktor u mnie zbagatelizował niepokojące objawy jakie wystąpiły po operacji i bez badań dodatkowych - wypisał do domu.',
    'Telewizja trochę szwankuje ( jest jakaś internetowa i czasem się przycina ) Niestety od przyszłego roku podobno ma być drożej ze względu na remont.',
    'Nasz pokój znajdował sie w rogu budynku, mimo to słychać było każde kroki i rozmowy z korytarza i innych pokoi.',
    'Wydałem 150zl plus dojazd.',
    'Stara suszarka wisząca na haczyku przez zawiniety kabel, tez nie napawa optymizmem, zwłaszcza ze po włączeniu w powietrzu unosił się zapach spalenizny.',
    'NIE PENSJONAT, WIĘC GDZIE JEST SALA ŚNIADANIOWA JAK W MARIOCIE W WARSZAWIE ?!',
    'Wykonałam na polecenie p. doktora niezliczone ilości kosztownych badań z których nic nie wynikało, zażywałam leki hormonalne, które powinnam brać pod okiem specjalisty w szpitalu na domiar złego każda kolejna wizyta u pana doktora trwałą 10 minut - bez wykonanego badania lekarskiego.',
    'Brak lobby i miejsc do wypoczynku.',
    'Generalnie odradzam bo jeśli lekarz czegoś nie jest pewien powinien się umieć przyznać do tego a nie udawać, że wie to karygodne tak niestety było w moim przypadku.',
    'Cale szczescie ze to byla tylko jedna noc bo nie mam zamiaru powtarzac przygody z tym hostelem.',
    'Byliśmy szczesciarzami, widziałem ludzi którzy oczekiwali po 8 godzin snując sie w upale po hotelu.',
    'Także zanim kupicie to polskie badziewie to zastanówcie się kilka razy - 3 letnia gwarancja na ten aku to śmiech na sali jak również serwis gwarancyjny.',
    'Na moja uwagę, iż katar ostatnio trwał długo i przerodzil się w wysiekowe zapalenie ucha i prośbę by poleciła coś na przyspieszenie jego ustapieniu, usłyszałam że nie ma potrzeby na tym etapie i oczywiście zawsze może tak się zdarzyć że katar spowoduje komplikacje.',
    'Jedzenia bardzo mało i niesmaczne A między obiadem A kolacja nie ma co jeść.',
    'We wszystkich informatorach widnieje godzina 22. 00.',
    'Wejście na teren kempingu nie był w żaden sposób monitorowany, nie czułam się więc bezpiecznie zostawiając tam rowery i namiot.',
    'Moim zdaniem jak jechac na wczasy to lepiej cos dorzucic i miec swiety spokoj i good time a nie ogladac to kiche co tam sie odprawia.',
    'Właściciel nie zaproponował jakiegokolwiek alternatywnego rozwiązania i rekompensaty, nie otrzymaliśmy nawet rachunku.',
    'Z wad jeszcze wymienił bym zacinanie się Youtuba.',
    'U mnie zaczęło się od tego że przesyłka szła do serwisu 13 dni bo takiego sobie przewoźnika wybrali... ale to był dopiero początek!',
    'Mało cierpliwy i nie ma moim zdaniem dobrego podejścia do pacjentek.',
    'Niestety, nie wyleczył mnie do końca - brakło mu pomysłów.',
    'Gburowaty, bardzo obraża ludzi, zupełnie niepotrzebne i przykre słowa w stosunku do drugiego człowieka, który przychodzi po poradę.',
    'Ocena " Słaby " za to, że powiedziała że prostatitis jest nieuleczalne co jest nieprawdą.',
    'Dodatkowo tuż po naszym przybyciu do pokoju była zupełnie pusta i nie mogliśmy napić się nawet wody po długiej podróży.',
    'Prócz skibusów brak komunikacji.',
    'Wino rozcieczone woda Jedzenie monotonne.',
    'Personel zupełnie niekompetentny!',
    'Obiadokolacji nie polecam.',
    'Pokoje niezbyt ładne.',
    'O wszystko trzeba się samemu pytać.',
    'Wywiad zajął Jej dłużej niż sama rozmowa o problemie i jeszcze usłyszał em, ze byłem za długo.',
    'Hotel nie ma parkingu i autokar musiał stać bardzo daleko.',
    'Winien jest profil miski.',
    'Z niekompletnym zrostem lekarka stwierdziła iż jestem zdrowy i nie widzi przeciwwskazań do wykonywania nawet ciężkich prac fizycznych.',
    'Pani doktor niestety kompletnie nie potrafiła mnie znieczulić, przez co w sumie nietrudne leczenie przerodziło się w koszmar dla mnie i orkę na ugorze dla niej.',
    'Szelki to tylko same paski parciane z kawałkiem skaju na biodro.',
    'W basenie zimna woda, w jacuzzi też.',
    'Mi zupełnie nie przypadł do gustu.',
    'Szkoda jedynie że jest na ciepło, przez co nie działa po stuknięciu paznokciem, długopisem czy rysikiem.',
    'do hoteu prowadzą schody, a więc ciężkie walizki trzeba nosić samodzielnie - brak podjazdu albo windy.',
    'Obiadokolacja w cenie 50 zl od osoby straszna, dania podane zimne, ociekajace tluszczem.',
    'Otoczenie ośrodka smutne, niezadbane.',
    'Mam taki bidon i jestem rozczarowana.',
    'Np. Odmawia wykonania odpłatnie USG jajników, bo coś jej się ubzdurało.',
    'Cena nie adekwatna do komfortu jaki oferuje hotel.',
    'Niestety już pierwszego dnia w recepcji czekała na nas bardzo niemiła niespodzianka.',
    'Prolaktyna nadal była ponad normą, natomiast pani ordynator przy wypisie przepisała mi sterydy ( mówiła zeby nie czytać ulotki bo moge sie wystraszyc ), jakiś środek na łaknienie ( w związku z moją rzekomą anoreksją ) i branie w dalszym ciągu norprolaku ( jak się później okazało nie powinno się go brać dłużej niż pół roku ).',
    'Sale prelekcyjne porażka, były prowadzone równocześnie dwa szkolenia poproszono o wyłączenie mikrofonów bo wzajemnie się zagłuszaliśmy.',
    'Nie odbiera telefonów, mimo, że wcześniej każe siebie informować o stanie zdrowia dziecka.',
    'Po otwarciu drzwi pokoju uderzał nieprzyjemny zapach pleśni i wilgoci.',
    'Nie było by tyle nie przespanych nocy, wątpliwości i mniej stresu.',
    'Montaz tej kamerki na monitorze graniczy z cudem bo uchwyt który posiada jest tak zaprojektowany że nie nadaje sie do niczego, Aby ją stabilnie zamontowac zamówiłem rzepy do podklejenia bo inaczej jest nie stabilna i co chwila trzeba ja poprawiac bo się sama obraca razem z mocowaniem.',
    'Dodam, że przed wizytą zrobiłam już część badań, z których dobry, doświadczony lekarz potrafił by wysunąć jakiekolwiek, chociaż drobne wnioski.',
    'Generalnie od 9. 00 do 22. 00 - 23. 00 ( wtedy kończyły się ostatnie imprezy ) cały czas coś gra...',
    'Stanowczo odradzam',
    'z wizyty u tego lekarza wyszłam załamana.',
    'A wszystko to okraszone dość wysokimi wymaganiami sprzętowymi jeśli zamierzamy grać na pełnych detalach graficznych.',
    'Zdecydowanie jednak nie chciała bym znaleźć się na miejscu tych ludzi.',
    'Niestety widok był z widokiem na parking i na miejsce gdzie ludzie podczas imprezy wychodzą palić papierosy.',
    'Pytania podczas laborki potrafią byc ciężkie, można wylecieć na sam koniec laboratorium, ale z reguły 3, może 4 grupy bedą zapytane podczas całych zajęć.',
    'Szatnie na dole kolo basenu co może być kłopotliwe dla niektórych.',
    'Male pokoje w ktorych widac ząb czasu.',
    'Nakłada i rozprowadza się idealnie ale najmniejszy zaciek kończy się ciemną plamą która przy próbie zamalowania robi się czarna i niemożliwa do usunięcia, gdyż w pomalowanym miejscu zarówno po wyschnięciu jak i przed wciągana jest w drewno kolejna warstwa impregnatu co doprowadza do sytuacji w której miejsce takie z kolejną warstwą robi się coraz ciemniejsze aż do całkowicie czarnego.',
    'Pani doktor w trakcie usg stwierdziła że faktycznie jestem w ciąży a gdy obydwoje z mężem popłakaliśmy się ze szczęscia zapytała z oburzeniem " czego pani ryczy ? ".',
    'Sprzætanie bardzo kiepsko.',
    'Ocena ogólna : UNIKAC : x : x',
    'Nie polecam!!!!!!!!!!!!!!!!!!!',
    'Jest to jawna niesprawiedliwość.',
    'Znajomi zaproponowali, żebym zostawił bagaże i przeszedł sie do hotelu obok na co Jakiś buc ( domniemam że właściciel ) siedzący w holu powiedział, że w żadnym wypadku nie mogę zostawić bagaży - mam je zabierać i wyjść... co z największą przyjemnością uczyniliśmy.',
    'Łazienka ze zniszczonym brodzikiem, zatkany brodzik, brudne fugi na płytkach, kontakty nieprzykręcone do ściany po malowaniu, na korytarzu czuć palonymi papierosami w recepcji, W opisie hotelu jest opcja śniadań lecz na miejscu okazuje się, że ich nie ma.',
    'z dnia na dzien bez słowa wyjasnien odwołali moją rezerwacje a gdy dzwoniłam zeby wyjaśnić dlaczego taka sytuacja zaistniała włascicielka była oprysklliwa i niemiła mówiąc " ze ją to nie obchodzi gdzie teraz znajde nocleg to mój kłopot itp " Już z wczesniejszych rozmów telefonicznych dało sie odczuć że to niezbyt przyjazni ludzie... odniosłam wrażenie że tak naprawde to łaskę mi robią że dają mi nocleg.',
    'Przyciski panelu sterowania głośno pikają po naciśnięciu.',
    'W pokoju przy łóżku nie ma żadnego gniazdka, żeby móc przez noc podładować telefon.',
    'Ten zestaw jest kompletnie do niczego.',
    'Po wejściu do gabinetu Pani Janiszewska nie pozwolił zamknąć drzwi wejściowych z tego powodu cała rozmowa była słyszana przez pacjentów na korytarzu.',
    'to sa te trafne zaściankowe diagnozy.',
    'Moje ostatnie dwie wizyty odbyły się niezgodnie z podaną wcześniej informacją i ostatecznie wyniosły mnie o 150 zł więcej niż byłam informowana wcześniej, a specjalnie pytałam kilka razy o koszt. .',
    'Zarówno poziomem wiedzy medycznej, jak i kultury osobistej.',
    'Pani dr nie stwierdziła nic niepokojącego a tym bardziej już podejrzenia tarczycy, bo to przecież ( mało prawdopodobne jej zdaniem ).',
    'Ale do dźwięku bym się przyczepił.',
    'Pająki w pokoju i szerszenie.',
    'A to w nieprzyjemny sposób " przesłuchiwał " na porannej wizycie śmiertelnie zmęczoną lekarkę przekazującą dyżur.',
    'Przypuszczam, że wszystkie jego pacjentki, które urodziły dzieci w trakcie leczenia u niego po prostu miały szczęście.',
    'Co jest najdziwniejsze ?',
    'Rozczarowała mnie również okolica.',
    'Nieprofesjonalna obsługa w barze ( pozdrowienia dla Dawida ), słaby DJ ( puszcza tytuły, a nie rytmy do tańczenia, nie rozumie co się do niego mówi ) nic dziwnego, że klub świeci pustkami, a jedyną sensowną atrakcją jest bowling w przesadnej cenie 80 zł / h ( jak rozumiem za dużo chętnych z powodów wymienionych wcześniej ) Kolejną sprawą jest sposób płatności na pokój, żeby nie nosić karty kredytowej lub gotówki np do basenu.',
    'Niestety miałam nieprzyjemność zetknąć się z nim znów w tej przychodni, gdy zgłosiłam się w trybie pilnym, z powodu zapalenia twardówki i błony naczyniowej oka.',
    'Zero w nich energii.',
    'Zimna woda ( jak sie odkreci prysznic na pol godziny, to leci ciepla ), lazienka na 2 pokoje ( w cenniku nazywaja to " studio " ), w pokoju zimno ( dostalem farelke, zeby sie dogrzac ), sniadanie kiepskie ( a calosc dopelnia kawa rozpuszczalna z automatu jak na dworcu ), Internet woooolny ( o ile akurat jest, bo rwie sie co chwile ).',
    'Jedyne, co mnie zaskoczyło, to pewne nieprzemyślane elementy budowy.',
    'Pokój z widokiem na północ 0 gór.',
    'Nie polecam korzystania z opcji obiadokolacji, zaserwowano zupę pieczarkową a na danie główne gołąbek z ziemniakami.',
    'Szkoda czasu i nerwów to rada dla wybierających się do profesora.',
    'Ogolnie butelki nie zostaly wysterelizowane, w toalecie w efekcie czego dziecko dostalo plesniawek.',
    'Niestety dzisiaj przyjechałam z mężem na wyczekiwanyw weekendw hotelu Diament i niestety już na samym początku okropne rozczarowanie - otrzymaliśmy pokój z widokiem na betonowy mur...',
    'Jednak pokój pozostawial wiele do życzenia.',
    'Również zmieszane zapachy wielu potraw nie zachęcały do dłuższego pobytu.',
    'Gondola i spacerówka to jest po prostu koszmar.',
    'Jak tylko dostaje zasilanie wpada w taki rezonans, że zagłusza dwa inne dyski + 2 wentylatory w obudowie, wentylator i buczenie zasilacza, chłodzenie procesora oraz dwa wentylatory grafiki razem wzięte.',
    'Jeśli ktoś liczy na basen do pływania z małymi dziećmi ( poniżej 4 lat ), musi wiedzieć że w hotelowym basenie nie mogą się kąpać dzieci.',
    'Ale jak juz sa komplikacje to chyba nie jest najlepszy.',
    'Oszczędzają ( a raczej dziadują ) na wszystkim, co się da.',
    'Po drugie : jedzenie było PASKUDNE!',
    'Żeby móc korzystać z pokoju musieliśmy go najpierw porządnie posprzątać.',
    'Kolejny raz i kolejne WIELKIE rozczarowanie, jeśli chodzi zarówno o czystość, obsługę, jedzenie i ogólne wrażenie.',
    'Jak na specjalistę w dziedzinie medycyny estetycznej, trochę slabe podejście do problemu.',
    'Badanie trwało dosłownie moment - dr nie zlecił natychmiastowego badania moczu, czego wymagał mój stan ( ból pochwy i pęcherza ).',
    'Jeden raz był makaron ( przypalony, z papryką chilli ).',
    'Na pewno nikomu nie polecam tego hotelu!',
    'Szklanki brudne stały nie zmieniane od niedzieli, sama je wynosilam i prosiłam o czyste, kabina prysznicowa nie myta.',
    'ZaleceniaKupując wczasy z biura Rainbow dobrze się zastanów czy umowa gwarantuje komfort jakiego oczekujesz sugerując się standardem pokoju w hotelu.',
    'nie warto wydawać pieniądze na niepotrzebne leki i tracic czas.',
    'Udałem sie do toalety gdzie zauważyłem ze zabieg spowodował krwawienie.',
    'Dodatkowo był niepomocny, niezainteresowany wyjaśnieniem problemu.',
    'Otrzymałem stanowczą odpowiedź, że mogę tylko otrzymać badania zgodne z dokumentacją pacjenta.',
    'Mała po tym dostała jeszcze większych kolek.',
    'Jedyny minus to bardzo wysokie ceny jak na Polskę 700 plan za nocleg, 1700 pln za 4 - osobowa salkę, 180 pln za steka z antrykotu z kieliszkiem wina, czy piwo za 20pln.',
    'Oczywiście nie obwiniam Doktora Hodery o to co się stało, bo to było " dziełem " natury, ale bardzo nie spodobało mi się Jego podejście do utracenia ciąży.',
    'Mało przyjemne.',
    'Może wystrój nie do końca w moim guście.',
]


positives = positives[:10]
negatives = negatives[:10]

def evaluate(dataset):
    correct = 0
    for txt, gold in dataset:
        pred = predict(txt)
        correct += int(pred == gold)
        print(f"[{gold}] {txt}\n -> pred: {pred}\n")
    acc = correct / len(dataset)
    print(f"Accuracy: {acc*100:.1f}%  ({correct}/{len(dataset)})")
    return acc

if __name__ == "__main__":
    # Sklej przykładowy zbiór z etykietami
    data = [(x, "pos") for x in positives] + [(x, "neg") for x in negatives]
    evaluate(data)
