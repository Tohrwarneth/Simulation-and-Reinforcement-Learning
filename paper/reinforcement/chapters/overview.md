# Erweiterung der Simulation

## Rückblick auf die aktuelle Simulation

Die Simulation wurde in drei Elementen aufgeteilt: der Fahrstuhlsteuerung, Personensteuerung und der verbindenden
Simulation. Dabei wird in jedem Takt überprüft, ob Personen einen Fahrstuhl zu einer Etage rufen. Ist der Fahrstuhl
leer, prüft er in jedem Takt, ob er gerufen wurde. Hat er jedoch Passagiere, so fährt er zur nächsten Zieletage und
nimmt, wenn möglich, auf dem Weg weitere Passagiere mit.

## Die Entscheider-Schnittstelle

Als erste Erweiterung wurde eine Entscheider-Schnittstelle erstellt, die von der Fahrstuhlsteuerung aufgerufen wird.
Diese gibt in der reinen Simulation die Entscheidung der Scanning-Strategie zurück.