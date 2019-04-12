# Analyse der Daten aud bc-daten.csv

Folgende Tasks sollen in Python durchgeführt werden:

* komplette EDA des Datasets (mit Visualisierung)
    * wieviele NA sind vorhanden?
    * Falls NA vorhanden sind sollen diese ersetzt werden
    * Finden der für die Prediction am Besten geeigneten Features
    * Aufbau folgender Modelle (mit Angabe von rsquared, precision, recall, f1)
        * Linear Regression
        * Logistic Regression
        * LDA
        * SVM
        * knn
        * Naive Bayes
        * Random Forrest
        * XGBoost
    * Confusion Matrix pro Modell
    * Wahl des besten Modells

## Attribute Information:

* ID number 
* Diagnosis (M = malignant, B = benign) 3-32)

Ten real-valued features are computed for each cell nucleus:

* radius (mean of distances from center to points on the perimeter) 
* texture (standard deviation of gray-scale values) 
* perimeter 
* area 
* smoothness (local variation in radius lengths) 
* compactness (perimeter^2 / area - 1.0) 
* concavity (severity of concave portions of the contour)
* concave points (number of concave portions of the contour) 
* symmetry 
* fractal dimension ("coastline approximation" - 1)

The mean, standard error and "worst" or largest (mean of the three largest values) 
of these features were computed for each image, resulting in 30 features. For instance, 
field 3 is Mean Radius, field 13 is Radius SE, field 23 is Worst Radius.

All feature values are recoded with four significant digits.

Class distribution: 357 benign, 212 malignant

## Analyse

Anzahl der Datensätze: 569

357 benign B

212 malignant M 

### Wieviele NA sind vorhanden ?
Es gibt keine Nullwerte. Details siehe _py/analyse_na.py_.

### Falls NA vorhanden sind sollen diese ersetzt werden ?
Falls NAs vorhanden wären könnten:
* Die entsprechenden Zeilen entfernt werden. Nachteil: Es sind nur wenig Testdaten vorhanden. Sollen wir da noch ganze Zeilen löschen ?
* Entsprechende Werte durch den Mittelwert der betroffnen Spalte ersetzt werden.

### Finden der für die Prediction am Besten geeigneten Features
####Analysiere die Einzelnen Features in Abhängigkeit zur Klassifizeirung.

![](images/box_mean.png)
![](images/box_se.png)
![](images/box_worst.png)

Folgende Attribute zeigen eine ausgeprägte Abhängigkeit von der Diagnose
* radius
* perimiter
* area
* compactness
* concavity
* concave_points


Folgende Attribute zeigen eine mittlere Abhängigkeit von der Diagnose
* texture

Folgende Attribute zeigen eine geringe Abhängigkeit von der Diagnose
* smoothness
* symmetry
* fractal_dimension

Generell ist die Abhängigkeit bei der Gruppe 'worst' und 'mean' ausgeprägter als bei 'se'.

#### Abhängigkeiten der Features untereinander
![](images/corr_mean.png)
![](images/corr_se.png)
![](images/corr_worst.png)

Auffällig (aber nicht weiter verwunderlich) ist die Korrelation zwischen radius, permiter und area.

Weiters scheint ein Zusammenhang zwischen compactness und concavity zu bestehen.

### Aufbau folgender Modelle (mit Angabe von rsquared, precision, recall, f1)
* rsquared macht wenig Sinn für Classification Problem
#### Linear Regression
Macht wenig Sinn
#### Logistic Regression
Verwendete Technologie: sklearn.linear_model.LogisticRegression


#### LDA
#### SVM
#### knn
#### Naive Bayes
#### Random Forrest
#### XGBoost
