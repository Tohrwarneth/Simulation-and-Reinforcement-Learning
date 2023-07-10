# Reinforcement Learning Konzepte

Im Laufe des Projektes wurden zwei verschiedene Konzepte und Implementierungen verwendet. Übergreifend wurde sich für
die Proximal Policy Optimization (PPO) entschieden.

## Proximal Policy Optimization (PPO)

2017 führte OpenAI den Algorithmus ein und aufgrund seiner Benutzerfreundlichkeit und guten Leistung zum
Standardalgorithmus für verstärktes Lernen bei OpenAI geworden. PPO ist ein on-policy Algorithmus, der, wie
die meisten klassischen Reinforcement-Learning-Algorithmen, am besten durch ein dichtes Belohnungssystem lernt. OpenAI
definierte für die Zielfunktion ihres Algorithmus wie folgt:

\label{formel_OpenAI}
$$ L^{\text{CLIP}}(\theta) = \hat{E_t}[\textit{min}(r_t(\theta), 1 - \varepsilon, q + \varepsilon) \hat{A_t})] $$

- $\theta$ ist der Policy Parameter
- $\hat{E_t}$ bezeichnet die empirische Erwartung über Zeitschritte
- $r_t$ ist das Verhältnis der Wahrscheinlichkeit unter der neuen und der alten Policy
- $\hat{A_t}$ ist der geschätzte Advantage zur Zeit $t$
- $\varepsilon$ ist ein Hyperparameter, normalerweise 0.1 oder 0.2

Diese Formel wird vor allem in der Implementierung in Kapitel \ref{impl_phill} genutzt.

## Grundkonzept vor der Implementierung

Zunächst wurde sich ein Lernziel definiert, welches die Maximierung der Reward-Funktion vorsieht. So sollen alle
Personen am Ende des simulierten Tages zu Hause sein und im Gesamten die Wartezeit minimiert werden.

Um dies zu erreichen, muss der Reinforcement Learner Beobachtungen in der simulierten Welt machen und diese in der
Auswertung der trainierten Iteration einfließen lassen.

### Weltbeobachtungen des Reinforcement Learners \label{concept_input}
Im ersten Konzept wurden daher folgende Parametern als Beobachtung dem Algorithmus weitergegeben: Die Zeit in Takten,
Anzahl der Personen im Aufzug, der aktuelle Steuerungszustand (Hoch, Runter, Warten) des Fahrstuhls, auf welche Etagen
ein Fahrstuhl in welche Richtung angefordert wurde und zuletzt die Zieletagen der Passagiere eines jeden Fahrstuhls.

Später wurden noch die durchschnittliche Wartezeit, die verbleibenden Personen im Gebäude und für jeden Aufzug die
Fahrtrichtung und Position in der Beobachtung ergänzt.

### Erste Reward-Funktion

Als Reward sollte zunächst die durchschnittliche Wartezeit prozentual zur Tageslänge errechnet werden. Dies würde mit
der maximalen Anzahl der Personen, die im Bürokomplex arbeiten, multipliziert werden, wenn keiner mehr im Büro ist.
Ansonsten wäre der Reward 0. Da sich jedoch über die Zeitschritte in der Simulation hinweg nur mit sehr viel Glück ein
Reward ergeben würde, und somit der Reinforcement Learner etwas zum Optimieren hätte, musste die Reward-Funktion
angepasst werden.

In der ersten Implementierung wurde die aktuelle durchschnittliche Wartezeit als negativer und Personen, die nach Hause
gingen, als positiver Reward übergeben. Dazu später in Kapitel \ref{phill_reward}.