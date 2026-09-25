# Universal Time-Series Forecasting: Akademische Dokumentation zur Vorhersage von Umweltzeitreihen

**Autor:** Omid Abduli
**Datum:** 2024
**System:** Universal Time-Series Forecasting

---

## 1. Zusammenfassung (Abstract)

Diese Dokumentation beschreibt die theoretischen, algorithmischen und softwaretechnischen Grundlagen von *Universal Time-Series Forecasting*. Das System wurde primär für Umweltwissenschaftler und Forscher entwickelt, um komplexe, nicht-lineare Zeitreihen-Szenarien vorherzusagen, insbesondere im Bereich der limnologischen (Seenforschung) und ökologischen Simulation.

Der Kern der Applikation ist eine automatisierte Machine-Learning-Pipeline (AutoML), die Datenbereinigung, Feature Engineering, Hyperparameter-Optimierung mittels Extreme Gradient Boosting (XGBoost) und Modell-Erklärbarkeit durch SHAP (SHapley Additive exPlanations) integriert. Die Architektur ist so konzipiert, dass sie ohne tiefergehende Programmierkenntnisse des Endnutzers operiert, während die zugrunde liegenden Berechnungen methodisch nachvollziehbar bleiben.

---

## 2. Einleitung

Die Modellierung von Umweltsystemen ist aufgrund der hohen Dimensionalität, Autokorrelation und Nicht-Stationarität der beteiligten Parameter (z.B. Wassertemperatur, gelöster Sauerstoff, pH-Wert) eine komplexe Herausforderung. Traditionelle deterministische Modelle (wie hydrodynamische Seenmodelle) erfordern oft detaillierte physikalische Parametrisierungen. Im Gegensatz dazu nutzt *Universal Time-Series Forecasting* einen datengetriebenen Ansatz (Data-Driven Modeling), bei dem Algorithmen die zugrunde liegenden Dynamiken direkt aus historischen Beobachtungen lernen.

Diese Dokumentation legt den Fokus auf die mathematischen Grundlagen und die algorithmische Implementierung der Pipeline.

---

## 3. Datenvorverarbeitung und Qualitätssicherung (Data Quality Assurance)

Bevor Algorithmen auf die Daten angewendet werden können, müssen diese bereinigt werden.

### 3.1 Identifikation von Fehlwerten (Missing Values)
Umweltdatensätze enthalten häufig systematische Platzhalter für Sensorausfälle (z.B. `-999`, `-9999`). Wenn ein Wert $x_i$ exakt einem bekannten Platzhalter entspricht, wird er maskiert:
$$ x_i = \text{NaN} \quad \forall \ x_i \in \{-999, -9999, 999, 9999\} $$

### 3.2 Detektion statistischer Ausreißer (Outlier Detection)
Die Erkennung von Ausreißern erfolgt über den Interquartilsabstand (Interquartile Range, IQR), um die Anfälligkeit gegenüber extremen Anomalien zu reduzieren, die bei der Nutzung der Standardabweichung auftreten würden.
Sei $Q_1$ das 25. Perzentil und $Q_3$ das 75. Perzentil der Verteilung eines Features $F$.
$$ \text{IQR} = Q_3 - Q_1 $$
Ein Datenpunkt $x$ gilt als potenzieller Ausreißer, wenn:
$$ x < Q_1 - \kappa \cdot \text{IQR} \quad \lor \quad x > Q_3 + \kappa \cdot \text{IQR} $$
Der Schwellenwert $\kappa$ (in der UI anpassbar, standardmäßig $3.0$) bestimmt die Striktheit der Filterung. 

### 3.3 Zeitreihen-Normalisierung
Die Zeitachse wird als Pandas `DatetimeIndex` formatiert. Eine chronologische Sortierung ist zwingend erforderlich, da gleitende Durchschnitte andernfalls über falsche Zeiträume gebildet würden.

---

## 4. Automatisches Feature Engineering

Zeitreihen enthalten strukturelle Informationen in ihrer Historie. Da Baumbasierte Modelle (wie XGBoost) standardmäßig keine zeitliche Sequenz erkennen (anders als LSTMs oder RNNs), müssen zeitliche Abhängigkeiten explizit als Features modelliert werden.

### 4.1 Rolling Window Statistics (Gleitende Statistiken)
Um das Signal-Rausch-Verhältnis zu verbessern und makroskopische Trends abzubilden, berechnet das Modul den gleitenden Durchschnitt (Moving Average) über ein Fenster der Größe $W$:
$$ \text{MA}_W(x^{(j)}_t) = \frac{1}{W} \sum_{i=0}^{W-1} x^{(j)}_{t-i} $$
Es werden standardmäßig gleitende Durchschnitte für 3 und 7 Zeitschritte gebildet (z.B. wöchentliche Glättung bei täglichen Daten). Im erweiterten Modus sind zusätzlich 14 und 30 Zeitschritte möglich. Die Zielvariable selbst wird nicht als Feature verwendet.

---

## 5. Mathematische Grundlagen des XGBoost Algorithmus

Das Herzstück der Vorhersage ist der **Extreme Gradient Boosting (XGBoost)** Regressor. Er gehört zur Familie der Ensemble-Lernverfahren und baut iterativ eine Sequenz von schwachen Lernenden (Decision Trees) auf.

### 5.1 Zielfunktion und Regularisierung
Die Zielfunktion $\mathcal{L}$ für Iteration $t$ (beim Hinzufügen des $t$-ten Baumes $f_t$) besteht aus einer Verlustfunktion $l$ und einem Regularisierungsterm $\Omega$:
$$ \mathcal{L}^{(t)} = \sum_{i=1}^n l(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) + \Omega(f_t) $$
Wobei $\hat{y}_i^{(t-1)}$ die Vorhersage des Modells aus dem vorherigen Schritt ist und $y_i$ der wahre Zielwert.
Für Regression wird häufig der mittlere quadratische Fehler (MSE) als Verlustfunktion $l$ genutzt.

Die Regularisierung des Baumes $f_t$ ist definiert als:
$$ \Omega(f_t) = \gamma T + \frac{1}{2} \lambda \sum_{j=1}^T w_j^2 + \alpha \sum_{j=1}^T |w_j| $$
Dabei ist $T$ die Anzahl der Blätter im Baum, $w_j$ das Gewicht (die Vorhersage) am Blatt $j$, $\gamma$ die Komplexitätsstrafe (reg_alpha im Code oft kombiniert), $\lambda$ (L2-Regularisierung) und $\alpha$ (L1-Regularisierung).

### 5.2 Taylor-Approximation
Um die Zielfunktion effizient zu minimieren, nutzt XGBoost eine Taylor-Entwicklung zweiter Ordnung:
$$ \mathcal{L}^{(t)} \approx \sum_{i=1}^n \left[ l(y_i, \hat{y}^{(t-1)}) + g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i) \right] + \Omega(f_t) $$
Hierbei sind $g_i$ (Gradient) und $h_i$ (Hesse-Matrix) die erste und zweite Ableitung der Verlustfunktion bezüglich der bisherigen Vorhersage:
$$ g_i = \partial_{\hat{y}^{(t-1)}} l(y_i, \hat{y}^{(t-1)}) $$
$$ h_i = \partial_{\hat{y}^{(t-1)}}^2 l(y_i, \hat{y}^{(t-1)}) $$

### 5.3 Optimales Baumwachstum
Für eine feste Baumstruktur $q(x)$ kann das optimale Blattgewicht $w_j^*$ berechnet werden als:
$$ w_j^* = - \frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda} $$
Der maximale Gewinn (Gain) bei einem Split (Knotenteilung) in eine linke ($L$) und rechte ($R$) Menge ist:
$$ \text{Gain} = \frac{1}{2} \left[ \frac{(\sum_{i \in I_L} g_i)^2}{\sum_{i \in I_L} h_i + \lambda} + \frac{(\sum_{i \in I_R} g_i)^2}{\sum_{i \in I_R} h_i + \lambda} - \frac{(\sum_{i \in I} g_i)^2}{\sum_{i \in I} h_i + \lambda} \right] - \gamma $$

---

## 6. Live-Optimierung der Hyperparameter

Das Programm führt eine stochastische (zufällige) Suche im Hyperparameter-Raum durch, um das Modell dynamisch an das hochgeladene Szenario anzupassen. Dies geschieht in der grafischen "Racing Chart" Oberfläche.

Folgende Parameter werden iterativ über $N$ Epochen optimiert:
- **Learning Rate ($\eta$):** Skaliert den Beitrag jedes neuen Baumes ($0.01 \leq \eta \leq 0.2$). Eine kleinere Lernrate verhindert Overfitting, benötigt aber mehr Bäume.
- **Max Depth:** Maximale Tiefe eines Baumes ($3 \leq d \leq 8$). Reguliert die Komplexität der Interaktionen zwischen Features.
- **N Estimators ($T_{total}$):** Anzahl der Bäume (Boosting-Runden) im Ensemble ($100 \leq T \leq 500$).

Die Zielfunktion der Optimierung ist die Maximierung des Determinationskoeffizienten ($R^2$) auf den Trainingsdaten (oder idealerweise auf einem Validierungs-Split).

---

## 7. Modellevaluierung (Scorecard Metriken)

Um die Leistung des Modells quantitativ zu erfassen, berechnet das Programm zwei zentrale Metriken.

### 7.1 Determinationskoeffizient ($R^2$)
Der $R^2$-Score misst den Anteil der Varianz in der abhängigen (Ziel-)Variablen, der durch das Modell aus den unabhängigen Variablen vorhergesagt werden kann.
$$ R^2 = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2} $$
- $\hat{y}_i$: Die Vorhersage des Modells.
- $\bar{y}$: Der arithmetische Mittelwert der wahren Daten.
Ein Wert von $1.0$ entspricht einer fehlerfreien Vorhersage. $0.0$ bedeutet, dass das Modell nicht besser abschneidet als eine einfache Mittelwertsvorhersage.

### 7.2 Root Mean Squared Error (RMSE)
Der RMSE quantifiziert die durchschnittliche absolute Abweichung der Vorhersagen. Da die Fehler quadriert werden, bestraft der RMSE große Abweichungen (Ausreißer) überproportional stark.
$$ \text{RMSE} = \sqrt{ \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2 } $$
Der RMSE ist dimensionsbehaftet und wird in denselben Einheiten wie die Zielvariable (z.B. Grad Celsius für Wassertemperatur) gemessen.

---

## 8. Erklärbare KI (Explainable AI): SHAP-Werte

Moderne Ensemble-Methoden werden oft als "Black Boxes" bezeichnet. Um dies zu lösen, implementiert *Universal Time-Series Forecasting* den SHAP-Ansatz (SHapley Additive exPlanations), basierend auf der kooperativen Spieltheorie von Lloyd Shapley.

### 8.1 Die Shapley-Formel
Der Shapley-Wert ordnet jedem Feature (Spieler) einen Beitrag (Auszahlung) an der Vorhersage (Gewinn) zu. Für ein Feature $j$ und ein Modell $f$ wird der SHAP-Wert $\phi_j$ berechnet durch das Marginalprodukt von $j$ bezogen auf alle möglichen Feature-Teilmengen $S$ (die $j$ nicht enthalten):
$$ \phi_j(x) = \sum_{S \subseteq F \setminus \{j\}} \frac{|S|! (|F| - |S| - 1)!}{|F|!} \left[ f_x(S \cup \{j\}) - f_x(S) \right] $$
Wobei:
- $F$: Die Menge aller Features.
- $f_x(S)$: Die erwartete Modellvorhersage konditioniert auf die Teilmenge $S$.

### 8.2 TreeExplainer
Da die exakte Berechnung exponentielle Laufzeit $\mathcal{O}(2^{|F|})$ hat, verwendet das Programm den `TreeExplainer`, einen optimierten Algorithmus von Lundberg et al. Dieser nutzt die interne Pfad-Struktur der XGBoost-Bäume, um die exakten SHAP-Werte in polynomieller Zeit $\mathcal{O}(T L D^2)$ zu berechnen ($T$=Bäume, $L$=Blätter, $D$=Tiefe).

Das Dashboard visualisiert den globalen Einfluss durch Aggregation der absoluten SHAP-Werte:
$$ I_j = \frac{1}{n} \sum_{i=1}^n |\phi_j(x_i)| $$
Dies beantwortet die Frage: *Welche Umweltparameter hatten im Durchschnitt den größten kausalen Einfluss auf die KI-Vorhersage?*

---

## 9. Softwarearchitektur und UI-Design

### 9.1 Modulare Architektur
- `app.py`: Zuständig für Orchestrierung, State-Management und Rendering des Streamlit-Frontends.
- `modules/data_loader.py`: Parser für komplexe Excel-Arbeitsmappen und Qualitätssicherung.
- `modules/feature_engineering.py`: Transformation der rohen Zeitreihenmatrix in einen hochdimensionalen Feature-Vektor.
- `modules/model.py`: XGBoost-Kapselung.
- `modules/explainer.py`: Interfacing mit der C-basierten SHAP-Bibliothek.
- `modules/visualizer.py`: Plotly-Generator für interaktive Vektorgrafiken.
- `modules/logger.py`: Serialisierung aller Simulations-Metadaten in persistente JSON-Artefakte für maschinelle Auswertung.

### 9.2 Oberfläche
Die Oberfläche ist eine Streamlit-App mit eigenem CSS. Während der Hyperparameter-Suche zeigt eine Live-Ansicht jeden getesteten Kandidaten, den Fortschritt der Validierung und den bisher besten Wert.

---

## 10. Schlussfolgerung

Universal Time-Series Forecasting macht maschinelles Lernen für Fachleute ohne Programmierkenntnisse nutzbar, zum Beispiel für Ökologen und Limnologen. Durch die Automatisierung des Feature Engineerings, dynamische Modelltuning-Routinen und die Erklärung der Entscheidungsfindung mittels spieltheoretischer SHAP-Werte bietet das System eine vollständige End-to-End-Pipeline für die prädiktive Umweltanalyse.

**Entwickelt und gestaltet von Omid Abduli.**
