# Erweiterung der Simulation

## Rückblick auf die aktuelle Simulation

Die Simulation wurde in drei Elementen aufgeteilt: der Fahrstuhlsteuerung, Personensteuerung und der verbindenden
Simulation. Dabei wird in jedem Takt überprüft, ob Personen einen Fahrstuhl zu einer Etage rufen. Ist der Fahrstuhl
leer, prüft er in jedem Takt, ob er gerufen wurde. Hat er jedoch Passagiere, so fährt er zur nächsten Zieletage und
nimmt, wenn möglich, auf dem Weg weitere Passagiere mit.

## Die Entscheider-Schnittstelle

Als erste Erweiterung wurde eine Entscheider-Schnittstelle erstellt, die von der Fahrstuhlsteuerung aufgerufen wird.
Diese gibt in der reinen Simulation die Entscheidung der Scanning-Strategie zurück. Ist der Entscheider des
Reinforcement Learnings aktiviert, gibt diese Schnittstelle eine neutrale Entscheidung zurück.
Die Entscheidung der Reinforcement Learners wurde bei der ersten Implementierung nach dem Fahrstuhlhandlings angewendet
(Abbildung \ref{elevator_RL}).

![Der neue abgeänderte Ablauf der Fahrstuhlsteuerung. Entscheider (grün), Neue Schritte (rot), Nicht mehr genutzt (lila)
\label{elevator_RL}](../images/elevator_RL.png)

In einer zweiten Implementierung wurde die Möglichkeit hinzugefügt, die Simulation nur Schrittweise auszuführen und beim
Training die Entscheidung außerhalb der Simulation zu treffen und diese für den nächsten Takt zu übergeben.
Ist nur der Entscheider dieser Reinforcement-Implementation genutzt, so kann die Simulation diese in jedem Takt
selbstständig vom Netz anfragen.