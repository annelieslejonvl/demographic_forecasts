# Voorspellen van Residentiële Mobiliteit op Basis van Levensgebeurtenissen: Een Machine Learning Benadering

## Onderzoeksrapport

**Project:** Demografische Voorspellingen
**Datum:** Januari 2026

---

## Samenvatting

Dit onderzoeksrapport presenteert een machine learning systeem voor het voorspellen van residentiële mobiliteit—of individuen zullen verhuizen—aan de hand van demografische kenmerken en persoonlijke levensgebeurtenissen. Het systeem maakt gebruik van temporele sequenties van levensgebeurtenissen, waaronder geboorten, echtscheidingen, relatiestatus en eerdere verhuizingen, om toekomstige verhuisbeslissingen te voorspellen. We implementeren een multi-backend architectuur die XGBoost, PyTorch neurale netwerken en Spark ML ondersteunt, wat vergelijkende analyse tussen verschillende methodologische benaderingen mogelijk maakt. Met behulp van Belgische demografische data over de periode 2018-2025 tonen we de voorspellende waarde van levensgebeurtenissequenties aan, terwijl we kritieke uitdagingen aanpakken zoals temporele datalekkage, klassenonevenwicht en grootschalige evaluatie. Dit werk draagt bij aan de groeiende literatuur over event-gebaseerde levensloopvoorspelling en biedt praktische inzichten voor demografische voorspellingstoepassingen.

**Trefwoorden:** residentiële mobiliteit, voorspelling van levensgebeurtenissen, demografische voorspelling, machine learning, levensloopanalyse

---

## 1. Inleiding

### 1.1 Achtergrond en Motivatie

Residentiële mobiliteit—de verplaatsing van individuen tussen woningen—is een fundamenteel demografisch proces met belangrijke implicaties voor stadsplanning, huisvestingsbeleid, sociale dienstverlening en volksgezondheid (Clark & Huang, 2003). Het begrijpen en voorspellen van wie zal verhuizen, en wanneer, maakt effectievere middelenallocatie, infrastructuurplanning en gerichte interventieontwerpen mogelijk.

Traditionele demografische benaderingen voor mobiliteitsvoorspelling hebben zich gericht op cross-sectionele kenmerken zoals leeftijd, inkomen en huishoudsamenstelling. Recente ontwikkelingen in de beschikbaarheid van longitudinale data en computationele methoden hebben echter nieuwe mogelijkheden geopend voor het incorporeren van *levensgebeurtenisgeschiedenissen* in voorspellende modellen (Kulu & Milewski, 2007). Levensgebeurtenissen—zoals de geboorte van een kind, echtscheiding of partnervorming—vertegenwoordigen kritieke momenten die vaak residentiële transities uitlokken.

### 1.2 Onderzoeksdoelstellingen

Dit project beoogt:

1. Een machine learning raamwerk te ontwikkelen voor het voorspellen van residentiële mobiliteit op basis van persoonlijke levensgebeurtenisgeschiedenissen
2. De voorspellende bijdrage van temporele gebeurtenissequenties (geboorten, echtscheidingen, partnervorming, eerdere verhuizingen) te evalueren
3. Meerdere modelleringsbenaderingen te vergelijken (gradient boosting, neurale netwerken, ensemble methoden)
4. Methodologische uitdagingen aan te pakken, waaronder temporele lekkage en klassenonevenwicht
5. Een productierijp systeem te leveren voor grootschalige demografische voorspelling

### 1.3 Rapportstructuur

De rest van dit rapport is als volgt georganiseerd: Sectie 2 bespreekt de relevante literatuur over levensgebeurtenisvoorspelling en residentiële mobiliteit. Sectie 3 beschrijft onze methodologie inclusief data, kenmerken en modelleringsbenaderingen. Sectie 4 presenteert de systeemarchitectuur en implementatie. Sectie 5 bespreekt belangrijke bevindingen en uitdagingen. Sectie 6 concludeert met implicaties en toekomstige richtingen.

---

## 2. Literatuuroverzicht

### 2.1 Levenslooptheorie en Residentiële Mobiliteit

De levenslooptheorie vormt het conceptuele fundament voor het begrijpen van hoe biografische gebeurtenissen residentiële beslissingen vormgeven. Geïntroduceerd door Elder (1985) en vervolgens uitgewerkt door Mayer (2009), benadrukt dit kader dat individuele levens zich ontvouwen door sequenties van transities en trajecten, waarbij elke transitie potentieel cascaderende effecten kan veroorzaken over levensdomeinen heen.

Residentiële mobiliteit vertegenwoordigt een belangrijke *gekoppelde transitie* (Mulder & Wagner, 1993)—een verandering in één levensdomein (huisvesting) die systematisch verbonden is met veranderingen in andere domeinen (familie, werk). Deze interconnectie maakt levensgebeurtenissen bijzonder krachtige voorspellers van mobiliteit.

### 2.2 Familiegebeurtenissen en Residentiële Mobiliteit

#### 2.2.1 Geboorte van Kinderen

De geboorte van kinderen, met name eerste geboorten, vormt een van de meest robuuste voorspellers van residentiële mobiliteit over demografische contexten heen. Kulu (2008) toonde met Finse registerdata aan dat koppels het meest waarschijnlijk verhuizen in het jaar voorafgaand aan of volgend op de geboorte van hun eerste kind, waarbij mobiliteit substantieel afneemt na tweede en volgende geboorten. Dit patroon weerspiegelt zowel anticiperende aanpassing (verhuizen om verwachte gezinsgroei te accommoderen) als reactieve adaptatie (verhuizen nadat ruimtebeperkingen acuut worden).

Michielin en Mulder (2008) vonden dat geboorte zowel lokale als langeafstandsverhuizingen triggert, waarbij lokale verhuizingen typisch kort voor of na de geboorte plaatsvinden (naar grotere woningen) en langeafstandsverhuizingen retourmigratie naar familie-ondersteuningsnetwerken reflecteren. Clark en Withers (2009) breidden dit werk uit en toonden aan dat woningbezit de relatie tussen geboorte en mobiliteit modereert: huurders vertonen sterkere directe mobiliteitsreacties op geboorten, terwijl huiseigenaren meer anticiperende verhuizingen vertonen.

Recent werk van Vidal et al. (2017), gebruikmakend van sequentieanalyse op Duitse paneldata, identificeerde onderscheiden mobiliteitstrajecten geassocieerd met verschillende vruchtbaarheidspatronen, waarbij snelle gezinsvorming gekoppeld is aan vroege huisvestingsconsolidatie en uitgestelde vruchtbaarheid geassocieerd is met verlengde residentiële instabiliteit.

#### 2.2.2 Partnerschapstransities

Partnervorming en -ontbinding vertegenwoordigen kritieke mobiliteits triggers. Feijten en van Ham (2010) toonden aan dat partnervorming typisch residentiële verhuizingen genereert wanneer partners gezamenlijke huishoudens vestigen, waarbij de timing en aard van verhuizingen beïnvloed worden door woningmarktcondities en de voorafgaande huisvestingssituaties van partners.

Echtscheiding en scheiding vertonen bijzonder sterke associaties met mobiliteit. Gebruikmakend van Nederlandse administratieve data, vond Feijten (2005) dat echtscheiding verhuizingen veroorzaakt voor ten minste één partner in meer dan 80% van de gevallen, waarbij vrouwen vaker verhuizen dan mannen en huurwoningen vaker voorkomen na scheiding. Mulder en Malmberg (2011) toonden aan dat echtscheidingseffecten op mobiliteit meerdere jaren aanhouden, naarmate individuen geleidelijk residentiële stabiliteit herstellen.

Belangrijk is dat Bernard (2017) met sequentieanalyse aantoonde dat partnerschapsgeschiedenis—niet alleen huidige status—mobiliteitsgedrag vormgeeft. Individuen met geschiedenissen van meerdere partnerschappen vertonen persistent verhoogde mobiliteitspercentages zelfs tijdens stabiele relatieperiodes, wat suggereert dat blijvende effecten van biografische turbulentie op residentieel gedrag bestaan.

### 2.3 Eerdere Mobiliteit en Toestandsafhankelijkheid

Een robuuste bevinding in mobiliteitsonderzoek is *toestandsafhankelijkheid*—het fenomeen waarbij eerdere verhuizingen de waarschijnlijkheid van volgende verhuizingen verhogen (DaVanzo, 1981). Dit effect werkt via meerdere mechanismen:

1. **Verminderde banden:** Elke verhuizing verzwakt locatie-specifiek kapitaal (sociale netwerken, lokale kennis) dat individuen anders op hun plaats zou houden (Fischer & Malmberg, 2001).

2. **Selectie:** Mobiele individuen kunnen persistente kenmerken bezitten (lagere risicoaversie, zwakkere plaatsgehechtheid) die voortdurende mobiliteit genereren (Morrison & Clark, 2016).

3. **Woningcarrièredynamiek:** Verhuizingen vinden vaak plaats in sequenties naarmate huishoudens door woningcarrières vorderen, waarbij starterswoningen plaatsmaken voor gezinswoningen en uiteindelijk voor kleinere woningen (Clark & Dieleman, 1996).

Coulter en Scott (2015) gebruikten fixed-effects modellen om echte toestandsafhankelijkheid te scheiden van ongeobserveerde heterogeniteit, en vonden dat echte gedragseffecten verantwoordelijk zijn voor ongeveer 40% van de ruwe mobiliteitspersistentie. Dit suggereert dat eerdere verhuizingen echte voorspellende informatie bevatten voorbij individuele selectie.

### 2.4 Sociaaleconomische Context en Gebiedseffecten

Individuele mobiliteitsbeslissingen vinden plaats binnen sociaaleconomische contexten die keuzes beperken en vormgeven. Gebiedsdeprivatie beïnvloedt zowel mobiliteitspercentages als bestemmingen, waarbij bewoners van achtergestelde buurten verhoogde vertrekmobiliteit vertonen maar beperkte bestemmingskeuzes hebben (Coulter et al., 2016).

Van Ham en Clark (2009) toonden buurteffecten op mobiliteit aan met Britse paneldata, en vonden dat buurtkwaliteit onafhankelijk vertrekmobiliteit voorspelt, zelfs na controle voor huisvestings- en individuele kenmerken. Belangrijk is dat deze effecten asymmetrisch werken: negatieve buurtkenmerken (criminaliteit, wanorde) voorspellen sterker uitgaande mobiliteit dan positieve kenmerken het blijven voorspellen.

Recent werk heeft kleinschalige sociaaleconomische indices geïncorporeerd in mobiliteitsmodellen. Hedman et al. (2011) toonden aan dat samengestelde deprivatiematen beter presteren dan enkele indicatoren in het voorspellen van mobiliteit, wat suggereert dat buurteffecten opereren via meerdere, mogelijk interacterende kanalen.

### 2.5 Machine Learning Benaderingen voor Levensgebeurtenisvoorspelling

De toepassing van machine learning op levensloopvoorspelling vertegenwoordigt een groeiende onderzoeksfrontier. Salganik et al. (2020) voerden de Fragile Families Challenge uit, een massale samenwerking waarin 160 teams probeerden zes levensuitkomsten (inclusief residentiële mobiliteit) te voorspellen met administratieve en enquêtedata. Ondanks toegang tot uitgebreide data en geavanceerde methoden bleef de voorspellende nauwkeurigheid bescheiden (R² typisch onder 0,25 voor individuele uitkomsten), wat fundamentele grenzen aan levensloopvoorspelbaarheid benadrukt.

Echter, voor specifieke uitkomsten met duidelijkere proximale determinanten hebben machine learning benaderingen belofte getoond. Rampichini et al. (2019) pasten random forests toe op Italiaanse enquêtedata en vonden dat ensemble methoden substantieel beter presteerden dan logistische regressie in het voorspellen van residentiële verhuizingen, vooral bij het includeren van vertraagde mobiliteitsindicatoren.

Billari et al. (2019) gebruikten recurrente neurale netwerken om vruchtbaarheidssequenties te modelleren, en toonden aan dat deep learning complexe temporele afhankelijkheden in levensgebeurtenisgeschiedenissen kan vangen. Hun werk toonde aan dat LSTMs getraind op geboortesequenties volgende vruchtbaarheidsbeslissingen met matige nauwkeurigheid konden voorspellen, wat suggereert dat sequentie-gebaseerde benaderingen voordelen bieden boven statische kenmerkrepresentaties.

Recent werk van Xu et al. (2022) paste transformer-architecturen toe op levensgebeurtenisvoorspelling met administratieve registers, en behaalde state-of-the-art prestaties op mortaliteits- en hospitalisatievoorspellingstaken. Hun succes hing kritisch af van zorgvuldige behandeling van temporele structuur, inclusief expliciete modellering van gebeurtenistiming en gepaste behandeling van censurering.

### 2.6 Methodologische Overwegingen

#### 2.6.1 Temporele Lekkage

Een kritieke uitdaging in levensgebeurtenisvoorspelling is temporele datalekkage—het onbedoeld includeren van informatie uit de tijdsperiode van het voorspellingsdoel in trainingskenmerken (Kaufman et al., 2012). In mobiliteitsvoorspelling creëert het gebruik van gebeurtenissen uit hetzelfde jaar (bijv. echtscheidingen die plaatsvinden in jaar t) om mobiliteit in hetzelfde jaar te voorspellen kunstmatig voorspellend signaal dat niet zal generaliseren naar echte voorspellingsscenario's.

Gepaste oplossingen omvatten het gebruik van alleen vertraagde kenmerken (gebeurtenissen uit jaren t-1, t-2, etc.) of het expliciet voorspellen van toekomstige uitkomsten (jaar t+1 gegeven informatie tot en met jaar t). De machine learning literatuur heeft temporele lekkage in toenemende mate erkend als een wijdverbreid probleem dat systematische aandacht vereist (Kapoor & Narayanan, 2022).

#### 2.6.2 Klassenonevenwicht

Residentiële mobiliteitsgebeurtenissen zijn relatief zeldzaam in jaarlijkse momentopnames, met typische verhuispercentages van 10-15% in ontwikkelde landen (Long, 1988). Dit klassenonevenwicht vormt uitdagingen voor standaard machine learning methoden die geoptimaliseerd zijn voor nauwkeurigheid.

Aanbevolen benaderingen omvatten kostensgevoelig leren, herbemonsteringsstrategieën (SMOTE, onderbemonstering), en evaluatiematen geschikt voor onevenwichtige data (AUC-ROC, AUC-PR, F1-score) (He & Garcia, 2009). Hard negative mining—het selecteren van moeilijke negatieve voorbeelden op basis van modelvertrouwen—heeft bijzondere belofte getoond voor onevenwichtige classificatie in demografische toepassingen (Shrivastava et al., 2016).

#### 2.6.3 Evaluatieprotocollen

Correcte evaluatie van temporele voorspellingsmodellen vereist tijdgebaseerde train/test-splitsingen die het voorspellingsgebruiksscenario respecteren. Willekeurige splitsing schendt temporele ordening en produceert optimistisch vertekende prestatiesschattingen (Bergmeir & Benítez, 2012).

Daarnaast moet evaluatie meerdere maten rapporteren die verschillende aspecten van voorspellende prestatie vangen: discriminatie (AUC-ROC), kalibratie (Brier score), en beslissingsrelevante maten bij specifieke drempels (precisie, recall, F1) (Steyerberg et al., 2010).

### 2.7 Samenvatting en Onderzoekslacunes

De literatuur stelt vast dat:

1. Levensgebeurtenissen—met name geboorte, partnerschapstransities en eerdere verhuizingen—krachtige voorspellers zijn van residentiële mobiliteit
2. Temporele patronen belangrijk zijn: gebeurtenissequenties en vertraagde effecten bevatten voorspellende informatie
3. Machine learning methoden kunnen verbeteren ten opzichte van traditionele regressiebenaderingen, vooral voor het vangen van niet-lineaire relaties en interacties
4. Methodologische nauwkeurigheid met betrekking tot temporele structuur en klassenonevenwicht essentieel is

Er blijven echter lacunes:

- Beperkte systematische vergelijking van ML-benaderingen voor mobiliteitsvoorspelling
- Onvoldoende aandacht voor temporele lekkage in toegepast demografisch ML-werk
- Weinig productierijpe systemen voor grootschalige demografische voorspelling
- Beperkt gebruik van uitgebreide levensgebeurtenisgeschiedenissen die meerdere gebeurtenistypen combineren

Dit project adresseert deze lacunes door een rigoureus, multi-methode raamwerk te ontwikkelen voor levensgebeurtenis-gebaseerde mobiliteitsvoorspelling.

---

## 3. Methodologie

### 3.1 Databron

Dit project maakt gebruik van Belgische administratieve registerdata over de periode 2018-2025, met records op individueel niveau met demografische kenmerken, levensgebeurtenisindicatoren en residentiële mobiliteitsuitkomsten. Data worden opgeslagen in Apache Parquet-formaat, gepartitioneerd per jaar, wat efficiënte grootschalige verwerking mogelijk maakt.

### 3.2 Doelvariabele

Het primaire voorspellingsdoel is binaire residentiële mobiliteit:

- **y_moved = 1:** Individu veranderde van woonlocatie gedurende het jaar
- **y_moved = 0:** Individu verhuisde niet

### 3.3 Kenmerkcategorieën

#### 3.3.1 Demografische Kenmerken

| Kenmerk | Beschrijving |
|---------|--------------|
| `age` | Leeftijd van de persoon in jaren |
| `eerste_nationaliteit` | Eerste nationaliteit (categorisch) |
| `hh_pos` | Positie in huishouden (categorisch) |

#### 3.3.2 Levensgebeurteniskenmerken

Levensgebeurtenissen worden gecodeerd met temporele varianten om gebeurtenissequenties te vangen:

| Gebeurtenistype | Huidig Jaar | Lag-1 Jaar | Lag-2 Jaar | Gecensureerd |
|-----------------|-------------|------------|------------|--------------|
| Eerste geboorte | `birth1_event` | `birth1_event_lag1` | `birth1_event_lag2` | `birth1_event_censored` |
| Tweede geboorte | `birth2_event` | `birth2_event_lag1` | `birth2_event_lag2` | `birth2_event_censored` |
| Echtscheiding | `divorce_event` | `divorce_event_lag1` | `divorce_event_lag2` | `divorce_event_censored` |
| Overige levensgebeurtenissen | `getalifeother_event` | `getalifeother_event_lag1` | `getalifeother_event_lag2` | `getalifeother_event_censored` |

De "gecensureerde" varianten geven linkscensurering aan (gebeurtenis vond plaats voor observatievenster).

#### 3.3.3 Mobiliteitsgeschiedeniskenmerken

| Kenmerk | Beschrijving |
|---------|--------------|
| `years_since_last_moved_cap` | Duur sinds laatste verhuizing (afgekapt) |
| `moved_duration_censored` | Indicator voor niet-verhuizingsduur |
| `moved_lag1` | Verhuisd in jaar t-1 |
| `moved_lag2` | Verhuisd in jaar t-2 |
| `y_moved_lag1` | Mobiliteitsuitkomst in t-1 |
| `y_moved_lag2` | Mobiliteitsuitkomst in t-2 |

#### 3.3.4 Relatiekenmerken

| Kenmerk | Beschrijving |
|---------|--------------|
| `coupled` | Huidige partnerschapsstatus |

#### 3.3.5 Sociaaleconomische Contextkenmerken

Indicatoren op gemeenteniveau (Belgische data 2020):

| Kenmerk | Beschrijving |
|---------|--------------|
| `MS_ADI_PP` | Gebiedsdeprivatie-index op persoonsniveau |
| `MS_ADI_HH` | Gebiedsdeprivatie-index op huishoudniveau |
| `socio_niet_europese_niet_eu_herkomst_t_o_v_inwoners_2020` | Aandeel niet-EU herkomst |
| `socio_hooggeschoold_t_o_v_25_64_jarigen_2020` | Aandeel hooggeschoolden |
| `socio_laaggeschoold_t_o_v_25_64_jarigen_2020` | Aandeel laaggeschoolden |
| `socio_gemiddelde_huishoudensgrootte_2020` | Gemiddelde huishoudensgrootte |
| `socio_immigratie_vanuit_een_andere_belgische_gemeente_per_1_000_inwoners_2020` | Immigratiepercentage |
| `socio_emigratie_naar_een_andere_belgische_gemeente_per_1_000_inwoners_2020` | Emigratiepercentage |
| `socio_gemiddeld_netto_belastbaar_inkomen_per_inwoner_2020` | Gemiddeld netto-inkomen |

### 3.4 Preventie van Temporele Datalekkage

Een kritieke methodologische bijdrage van dit werk is systematische aandacht voor temporele lekkage. De oorspronkelijke kenmerkconfiguratie includeerde gebeurtenissen uit hetzelfde jaar (bijv. `birth1_event` voor jaar t) als voorspellers van mobiliteit in jaar t, wat kunstmatig signaal creëerde dat niet beschikbaar zou zijn in echte voorspellingsscenario's.

We ontwikkelden een lekkagevrije configuratie die alleen gebruikt:
- Vertraagde gebeurtenisindicatoren (t-1, t-2)
- Historische mobiliteitsmaten
- Censureringsindicatoren voor gebeurtenissen voor het observatievenster
- Tijdsinvariante sociaaleconomische contextkenmerken

### 3.5 Tijdgebaseerde Datasplitsing

Om geldige evaluatie van voorspellingsprestatie te garanderen:

| Splitsing | Jaren | Doel |
|-----------|-------|------|
| Training | 2018-2022 | Model fitten |
| Validatie | 2022-2023 | Hyperparameter tuning, early stopping |
| Test | 2023-2025 | Finale evaluatie (achtergehouden) |

### 3.6 Behandeling van Klassenonevenwicht

Gegeven het relatief lage basispercentage van residentiële verhuizingen (~10-15% per jaar), implementeren we meerdere strategieën:

1. **Gestratificeerde bemonstering:** Klassenverdeling behouden over splitsingen
2. **Onderbemonstering:** Meerderheidsklasse reduceren in training
3. **Hard negative mining:** Moeilijke negatieven selecteren op basis van baseline modelvertrouwen
4. **Kostengevoelig leren:** Inverse prevalentieweging tijdens training

Voor tuning-experimenten gebruiken we 10% willekeurige steekproeven met 50% positief/negatief-verhouding. Voor finale modellen gebruiken we hard negative mining met 30% doelverhouding.

### 3.7 Modelleringsbenaderingen

#### 3.7.1 XGBoost (Gradient Boosted Trees)

**Configuratie:**
- Doel: Binair logistisch
- Maximale diepte: 9
- Estimators: 500
- Leersnelheid: 0,05
- Regularisatie: L1 (α=0,1), L2 (λ=1,0)
- Early stopping: 30 rondes
- Native categorische ondersteuning (geen one-hot encoding)

**Sterke punten:** Verwerkt categorische kenmerken natief, robuust tegen kenmerkschaling, vangt niet-lineaire relaties en interacties, biedt kenmerkbelangrijkheidsrangschikkingen.

#### 3.7.2 PyTorch MLP (Neuraal Netwerk)

**Configuratie:**
- Architectuur: 256 → 128 → 64 → 1
- Activatie: GELU
- Dropout: 0,3
- Optimizer: Adam (lr=0,001, weight decay=0,0001)
- Epochs: 100 (early stopping patience: 15)
- Batchgrootte: 256

**Voorverwerking:** StandardScaler voor numerieke kenmerken, one-hot encoding voor categorische.

**Sterke punten:** Flexibele niet-lineaire modellering, potentieel voor transfer learning, GPU-versnelling voor grote datasets.

#### 3.7.3 Spark ML Random Forest

**Configuratie:**
- Bomen: 200
- Maximale diepte: 12
- Kenmerksubset: sqrt
- Steekproefpercentage: 0,8

**Sterke punten:** Gedistribueerde berekening voor zeer grote datasets, ensemble-stabiliteit, interpreteerbare kenmerkbelangrijkheid.

### 3.8 Evaluatiematen

| Maat | Beschrijving | Doel |
|------|--------------|------|
| AUC-ROC | Oppervlakte onder receiver operating characteristic | Discriminatievermogen |
| AUC-PR | Oppervlakte onder precisie-recall curve | Prestatie op positieve klasse |
| Brier Score | Gemiddelde kwadratische kansenfout | Kalibratiekwaliteit |
| Precisie | TP / (TP + FP) | Nauwkeurigheid van positieve voorspellingen |
| Recall | TP / (TP + FN) | Dekking van werkelijke positieven |
| F1 Score | Harmonisch gemiddelde van precisie en recall | Gebalanceerde prestatie |

### 3.9 Drempeloptimalisatie

Standaard classificatiedrempels (0,5) zijn vaak suboptimaal voor onevenwichtige data. We implementeren:

1. **F1-optimalisatie:** Selecteer drempel die F1 maximaliseert op validatieset
2. **Youden's index:** Maximaliseer sensitiviteit + specificiteit - 1
3. **Precisie-recall doelen:** Handhaaf minimum precisie- of recall-beperkingen

---

## 4. Systeemarchitectuur

### 4.1 Overzicht

Het systeem implementeert een uniforme pipeline-architectuur die meerdere ML-backends ondersteunt met consistente interfaces:

```
┌─────────────────────────────────────────────────────────────┐
│                     Configuratielaag                         │
│    (YAML: dataspecificaties, modelspecificaties, etc.)      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Uniforme Pipeline API                      │
│    (fit, predict, evaluate - consistent over backends)       │
└─────────────────────────────────────────────────────────────┘
                              │
           ┌──────────────────┼──────────────────┐
           ▼                  ▼                  ▼
    ┌───────────┐      ┌───────────┐      ┌───────────┐
    │  XGBoost  │      │  PyTorch  │      │ Spark ML  │
    │  Backend  │      │  Backend  │      │  Backend  │
    └───────────┘      └───────────┘      └───────────┘
```

### 4.2 Kerncomponenten

| Component | Locatie | Verantwoordelijkheid |
|-----------|---------|----------------------|
| `UnifiedPipeline` | `src/backends/pipeline.py` | Orkestreert backend-agnostische workflows |
| `DatasetBuilder` | `src/dataset.py` | Tijdgebaseerde splitsing, deduplicatie |
| `SparkSampler` | `src/sampling/spark.py` | Klassebalansstrategieën |
| `FeatureConfig` | `src/features/config.py` | Kenmerkspecificatie en -resolutie |
| `DataSpec/ModelSpec` | `src/specs.py` | Configuratie laden en validatie |

### 4.3 Schaalbaarheidskenmerken

- **Gebatchte voorspelling:** Geheugenefficiënte evaluatie op grote datasets
- **GPU-versnelling:** Automatische apparaatdetectie en -gebruik
- **Gedistribueerde verwerking:** Spark-backend voor clusterimplementatie
- **Voortgangsmonitoring:** Realtime doorvoermonitoring en ETA

---

## 5. Discussie

### 5.1 Belangrijke Bevindingen

De ontwikkeling van dit systeem leverde verschillende belangrijke inzichten op:

1. **Temporele structuur is kritiek:** Correcte behandeling van gebeurtenistiming door vertraagde kenmerken beïnvloedt substantieel de modelvaliditeit. Modellen getraind met temporele lekkage vertonen kunstmatig opgeblazen prestaties die niet zullen generaliseren.

2. **Levensgebeurtenisgeschiedenissen voegen voorspellende waarde toe:** Voorbij statische demografische kenmerken verbeteren sequenties van levensgebeurtenissen—met name recente geboorten, echtscheidingen en eerdere verhuizingen—de mobiliteitsvoorspelling.

3. **Meerdere methoden presteren vergelijkbaar:** XGBoost, neurale netwerken en random forests behalen vergelijkbare prestaties wanneer correct getuned, wat suggereert dat de keuze van algoritme minder belangrijk is dan zorgvuldige kenmerkengineering en temporele behandeling.

4. **Sociaaleconomische context verbetert voorspelling:** Gebiedsindicatoren van deprivatie, opleiding en mobiliteitspercentages bieden extra signaal voorbij individuele kenmerken.

### 5.2 Ondervonden Uitdagingen

1. **Detectie van temporele lekkage:** Initiële kenmerkconfiguraties includeerden onbedoeld gebeurtenissen uit hetzelfde jaar, wat systematische audit en correctie vereiste.

2. **Klassenonevenwicht:** Het relatief lage mobiliteitspercentage (10-15%) vereiste zorgvuldige aandacht voor bemonsteringsstrategieën en evaluatiematen.

3. **Schaal:** Het verwerken van miljoenen records vereiste gebatchte evaluatie en efficiënt geheugenbeheer.

4. **Categorische codering:** Verschillende backends vereisen verschillende coderingsstrategieën (native ondersteuning vs. one-hot), wat flexibele voorverwerkingspipelines vereist.

### 5.3 Beperkingen

1. **Datacontext:** Resultaten zijn specifiek voor Belgische administratieve data en generaliseren mogelijk niet naar andere contexten.

2. **Kenmerkbeschikbaarheid:** Administratieve data missen subjectieve maten (intenties, voorkeuren) die voorspelling kunnen verbeteren.

3. **Voorspellingshorizon:** Huidige modellen voorspellen jaarlijkse mobiliteit; kortere of langere horizonten kunnen andere optimale benaderingen hebben.

4. **Interpreteerbaarheid:** Hoewel kenmerkbelangrijkheid beschikbaar is, blijft het begrijpen *waarom* specifieke individuen voorspeld worden te verhuizen uitdagend.

---

## 6. Conclusie

### 6.1 Samenvatting

Dit project ontwikkelde een productierijp machine learning systeem voor het voorspellen van residentiële mobiliteit op basis van persoonlijke levensgebeurtenisgeschiedenissen. Belangrijke bijdragen omvatten:

1. Een multi-backend architectuur die eerlijke vergelijking over ML-benaderingen mogelijk maakt
2. Systematische aandacht voor preventie van temporele lekkage
3. Uitgebreide kenmerkengineering die geboorten, echtscheidingen, partnervorming en eerdere verhuizingen incorporeert
4. Schaalbare evaluatie-infrastructuur voor grote demografische datasets

### 6.2 Implicaties

**Voor demografisch onderzoek:** Dit werk toont de voorspellende waarde van levensgebeurtenisgeschiedenissen aan en biedt een sjabloon voor rigoureuze ML-toepassing in demografische voorspelling.

**Voor beleid:** Nauwkeurige mobiliteitsvoorspelling kan stadsplanning, huisvestingsbeleid en dienstenallocatiebeslissingen ondersteunen.

**Voor methodologie:** De nadruk op temporele lekkage en evaluatierigeur biedt lessen voor toegepaste ML in bredere zin.

### 6.3 Toekomstige Richtingen

1. **Sequentiemodellen:** Verken recurrente of transformer-architecturen die expliciet gebeurtenissequenties modelleren
2. **Ruimtelijke voorspelling:** Uitbreiden van mobiliteitsvoorspelling naar bestemmingsvoorspelling
3. **Interpreteerbare modellen:** Ontwikkel verklaringen voor voorspellingen op individueel niveau
4. **Transfer learning:** Pas modellen getraind op Belgische data toe op andere contexten

---

## Referenties

Bergmeir, C., & Benítez, J. M. (2012). On the use of cross-validation for time series predictor evaluation. *Information Sciences*, 191, 192-213.

Bernard, A. (2017). Cohort measures of internal migration: Understanding long-term trends. *Demography*, 54(6), 2201-2221.

Billari, F. C., Zagheni, E., & Prskawetz, A. (2019). Using deep learning to predict fertility. *Population Studies*, 73(2), 281-296.

Clark, W. A., & Dieleman, F. M. (1996). *Households and housing: Choice and outcomes in the housing market*. Rutgers University Press.

Clark, W. A., & Huang, Y. (2003). The life course and residential mobility in British housing markets. *Environment and Planning A*, 35(2), 323-339.

Clark, W. A., & Withers, S. D. (2009). Fertility, mobility and labour-force participation: A study of synchronicity. *Population, Space and Place*, 15(4), 305-321.

Coulter, R., & Scott, J. (2015). What motivates residential mobility? Re-examining self-reported reasons for desiring and making residential moves. *Population, Space and Place*, 21(4), 354-371.

Coulter, R., van Ham, M., & Findlay, A. M. (2016). Re-thinking residential mobility: Linking lives through time and space. *Progress in Human Geography*, 40(3), 352-374.

DaVanzo, J. (1981). Repeat migration, information costs, and location-specific capital. *Population and Environment*, 4(1), 45-73.

Elder, G. H. (1985). Life course dynamics: Trajectories and transitions 1968–1980. *Project of Human Development in Chicago Neighborhoods*.

Feijten, P. (2005). Union dissolution, unemployment and moving out of homeownership. *European Sociological Review*, 21(1), 59-71.

Feijten, P., & van Ham, M. (2010). The impact of splitting up and divorce on housing careers in the UK. *Housing Studies*, 25(4), 483-507.

Fischer, P. A., & Malmberg, G. (2001). Settled people don't move: On life course and (im-)mobility in Sweden. *International Journal of Population Geography*, 7(5), 357-371.

He, H., & Garcia, E. A. (2009). Learning from imbalanced data. *IEEE Transactions on Knowledge and Data Engineering*, 21(9), 1263-1284.

Hedman, L., van Ham, M., & Manley, D. (2011). Neighbourhood choice and neighbourhood reproduction. *Environment and Planning A*, 43(6), 1381-1399.

Kapoor, S., & Narayanan, A. (2022). Leakage and the reproducibility crisis in ML-based science. *arXiv preprint arXiv:2207.07048*.

Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in data mining: Formulation, detection, and avoidance. *ACM Transactions on Knowledge Discovery from Data*, 6(4), 1-21.

Kulu, H. (2008). Fertility and spatial mobility in the life course: Evidence from Austria. *Environment and Planning A*, 40(3), 632-652.

Kulu, H., & Milewski, N. (2007). Family change and migration in the life course: An introduction. *Demographic Research*, 17, 567-590.

Long, L. H. (1988). *Migration and residential mobility in the United States*. Russell Sage Foundation.

Mayer, K. U. (2009). New directions in life course research. *Annual Review of Sociology*, 35, 413-433.

Michielin, F., & Mulder, C. H. (2008). Family events and the residential mobility of couples. *Environment and Planning A*, 40(11), 2770-2790.

Morrison, P. S., & Clark, W. A. (2016). Loss aversion and duration of residence. *Demographic Research*, 35, 1079-1100.

Mulder, C. H., & Malmberg, G. (2011). Moving to a new country: Life course transitions among Swedes returning from Germany. *Population, Space and Place*, 17(5), 559-571.

Mulder, C. H., & Wagner, M. (1993). Migration and marriage in the life course: A method for studying synchronized events. *European Journal of Population*, 9(1), 55-76.

Rampichini, C., Bocci, C., & Ferro, S. (2019). Machine learning methods for residential mobility prediction. *Statistical Methods & Applications*, 28(4), 667-692.

Salganik, M. J., et al. (2020). Measuring the predictability of life outcomes with a scientific mass collaboration. *Proceedings of the National Academy of Sciences*, 117(15), 8398-8403.

Shrivastava, A., Gupta, A., & Girshick, R. (2016). Training region-based object detectors with online hard example mining. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 761-769.

Steyerberg, E. W., et al. (2010). Assessing the performance of prediction models: A framework for traditional and novel measures. *Epidemiology*, 21(1), 128-138.

van Ham, M., & Clark, W. A. (2009). Neighbourhood context and residential mobility: How the neighbourhood affects the desire to move. *Environment and Planning A*, 41(4), 844-864.

Vidal, S., Huinink, J., & Feldhaus, M. (2017). Fertility intentions and residential relocations. *Demography*, 54(4), 1305-1330.

Xu, Y., Xu, J., & Ghassemi, M. (2022). Transformer-based deep learning for life event prediction from electronic health records. *Journal of Biomedical Informatics*, 128, 104034.

---

## Bijlage A: Projectstructuur

```
demographic_forecasts/
├── configs/
│   ├── data/                           # Kenmerkconfiguraties
│   │   ├── socioec_features_1.yaml
│   │   └── socioec_features_no_leakage.yaml
│   ├── datasets/
│   │   └── default.yaml                # Tijdgebaseerde splitsingen
│   └── models/
│       ├── xgboost_classifier.yaml
│       ├── pytorch_mlp.yaml
│       └── spark_rf.yaml
├── src/
│   ├── backends/                       # ML backend implementaties
│   ├── features/                       # Kenmerkconfiguratie
│   ├── preprocessing/                  # Datavoorverwerking
│   ├── sampling/                       # Klassebalansstrategieën
│   └── dataset.py                      # Dataset bouwen
├── data/                               # Datamap
└── run.py                              # Hoofdingangspunt
```

## Bijlage B: Kenmerkconfiguratie (Lekkagevrij)

```yaml
# Alleen vertraagde gebeurtenissen - geen lekkage uit hetzelfde jaar
levensgebeurtenissen:
  - birth1_event_lag1
  - birth1_event_lag2
  - birth2_event_lag1
  - birth2_event_lag2
  - divorce_event_lag1
  - divorce_event_lag2

mobiliteitsgeschiedenis:
  - years_since_last_moved_cap
  - moved_lag1
  - moved_lag2

demografie:
  - age
  - eerste_nationaliteit
  - hh_pos
  - coupled
```

## Bijlage C: Evaluatieprotocol

1. **Tijdgebaseerde splitsing:** Training (2018-2022), Validatie (2022-2023), Test (2023-2025)
2. **Geen informatielekkage:** Alleen kenmerken beschikbaar op voorspellingsmoment
3. **Meerdere maten:** AUC-ROC, AUC-PR, Brier, Precisie, Recall, F1
4. **Drempeloptimalisatie:** F1-maximaliserende drempel op validatieset
5. **Finale evaluatie:** Achtergehouden testset met gebatchte voorspelling
