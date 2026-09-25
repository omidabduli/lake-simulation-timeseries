# Universal Time-Series Forecasting: Sehr einfache Erklärung

**Autor:** Omid Abduli

Hallo! Dieser Text erklärt, was *Universal Time-Series Forecasting* macht. Wenn die mathematischen Formeln in der ausführlichen Dokumentation zu kompliziert sind, findest du hier eine leicht verständliche Einführung.

---

## 1. Was macht das Programm eigentlich?

Stell dir vor, du hast ein schlaues Notizbuch. Jeden Tag schreibst du auf, wie das Wetter ist: Temperatur, Wind, und ob es regnet. Nach ein paar Monaten willst du wissen: "Wie wird das Wetter morgen?" 

Dieses Programm (die KI) liest dein Notizbuch, lernt daraus und versucht dann, die Zukunft zu erraten. Wir nennen das **Zeitreihen-Vorhersage** (Time-Series Prediction).

---

## 2. Daten saubermachen (Data Cleaning)

**Das Problem:** Manchmal ist dein Notizbuch schmutzig. Vielleicht hast du an einem Tag vergessen, das Wetter aufzuschreiben und hast stattdessen "-9999" hingeschrieben. Oder du hast dich verschrieben und "1000 Grad Celsius" notiert!

**Was die KI macht:** 
Bevor die KI anfängt zu lernen, räumt sie das Zimmer auf. 
1. **Fehlwerte (Missing Values):** Sie sucht nach diesen "-9999" Zahlen und wirft sie weg, weil sie weiß, dass das keine echten Temperaturen sind.
2. **Ausreißer (Outliers):** Sie schaut sich an, was "normal" ist. Wenn die Temperatur normalerweise zwischen 10 und 30 Grad liegt, und plötzlich steht da 500 Grad, sagt die KI: "Moment mal, das ist ein Ausreißer (ein Fehler)!" und ignoriert diese Zahl.

---

## 3. Feature Engineering (Sich an gestern erinnern)

**Das Problem:** Wenn ich dich frage: "Wird es morgen regnen?", ist es super wichtig zu wissen, ob es *heute* geregnet hat. Die KI schaut sich normalerweise aber nur den aktuellen Tag an.

**Was die KI macht:**
Wir bringen der KI bei, sich zu erinnern! Wir bauen "Lags" (Verzögerungen) ein. Das bedeutet, wir geben der KI zusätzliche Zettel, auf denen steht: "Gestern war es so", "Vorgestern war es so" und "Vor einer Woche war es so". Das nennt man **Feature Engineering**. So kann die KI Muster erkennen (z.B. "Immer wenn es zwei Tage lang windig war, regnet es am dritten Tag").

---

## 4. Das Gehirn der KI: XGBoost (Das super schlaue Team)

Das "Gehirn" dieses Programms heißt **XGBoost**. Das klingt kompliziert, ist aber eigentlich eine tolle Teamarbeit.

**Die Analogie:**
Stell dir vor, du hast ein großes Glas voller Gummibärchen und fragst ein Kind im Kindergarten: "Wie viele sind da drin?"
1. Das erste Kind rät: "100!"
2. Es sind aber 150 drin. Also sagen wir dem zweiten Kind: "Das erste Kind lag um 50 daneben (zu wenig)."
3. Das zweite Kind rät nicht die Gesamtzahl, sondern versucht nur, den *Fehler* des ersten Kindes zu korrigieren. Es sagt: "Dann addiere ich 30!"
4. Jetzt sind wir bei 130. Wir sagen dem dritten Kind: "Wir liegen noch 20 daneben."
5. Das dritte Kind sagt: "Ich addiere 20!"

Genau das macht XGBoost! Es baut nicht einen riesigen schlauen Computer, sondern **Hunderte kleine, einfache Entscheidungsbäume** (die Kinder). Jeder neue Baum schaut sich nur die *Fehler* an, die alle Bäume vor ihm gemacht haben, und versucht, genau diesen Fehler zu beheben. Alle zusammen sind am Ende ein super schlaues Team!

---

## 5. Live-Optimierung (Das beste Team zusammenstellen)

Wenn das Programm "Training forecast engine" anzeigt, dann probiert es aus, wie das Team am besten arbeitet. 
- Brauchen wir 100 Kinder oder 500 Kinder? (N-Estimators)
- Sollen die Kinder kleine Korrekturen machen oder große Schritte wagen? (Learning Rate)
Das Programm probiert live ganz viele Kombinationen aus und behält am Ende das Team, das am besten geraten hat!

---

## 6. Noten vergeben: RMSE und R² (Das Zeugnis)

Wie wissen wir, ob die KI gut ist? Wir geben ihr Noten!

*   **R² (R-Quadrat):** Das ist wie die Schulnote. 1.0 (oder 100%) bedeutet eine glatte Eins! Die KI hat alles perfekt vorhergesagt. 0.0 bedeutet, die KI hat nur geraten.
*   **RMSE (Avg Error):** Stell dir wieder die Gummibärchen vor. Wenn die KI im Durchschnitt immer um 5 Gummibärchen daneben liegt, ist der RMSE = 5. Je kleiner diese Zahl ist, desto besser ist die KI!

---

## 7. Erklärbare KI: SHAP (Wer war der wichtigste Spieler?)

Oft sagen KIs einfach ein Ergebnis, und wir wissen nicht, *warum*. Unser Programm nutzt **SHAP**.

**Die Analogie:**
Stell dir vor, eine Fußballmannschaft gewinnt 3:0. Wir wollen wissen: Wer war der beste Spieler? Wer hat das Team zum Sieg geführt?
SHAP schaut sich das Spiel an und berechnet für jeden Spieler genau aus, wie wichtig er war. 

In unserem Programm sagt SHAP nicht, wer die besten Fußballspieler waren, sondern **welche Daten** am wichtigsten waren. Wenn die KI also eine Wassertemperatur vorhersagt, sagt uns SHAP: "Der wichtigste Grund für diese Vorhersage war die Lufttemperatur von gestern, danach kam der Wind, und der Regen war gar nicht so wichtig."

---

## Zusammenfassung

Das Programm nimmt schmutzige Daten, räumt sie auf (Cleaning), schaut sich die Vergangenheit an (Feature Engineering), lässt ein Team von hunderten kleinen Helfern aus ihren Fehlern lernen (XGBoost), benotet das Ergebnis (RMSE/R²) und verrät uns am Ende ganz genau, warum es sich so entschieden hat (SHAP). 

Alles läuft automatisch.
