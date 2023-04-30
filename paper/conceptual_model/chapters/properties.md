# Eigenschaften

[//]: # (Wie sieht die innere Struktur des SUIs aus?)

[//]: # (Wie sieht der Fahrstuhl aus)

## Innere Struktur des Fahrstuhls

Der Fahrstuhl ist in 3 Elementen aufgeteilt.

- Stockwerksposition und Fahrtrichtung
- Personenanzahl für nach oben und unten des aktuellen Stockwerks
- Innenraum

Der Innenraum zeigt, wie viele Personen sich in ihm befinden und wo
welche Knöpfe in welcher Reihenfolge gedrückt wurden.
Des Weiteren wird sein Fahrziel hervorgehoben angezeigt.

[//]: # (Welche Zustandsvariablen gibt es?)

[//]: # (Warten, Hoch, Runter)

Es gibt drei Zustände je Fahrstuhl:

- Warten
- Hoch
- Runter

Warten fasst zwei Zustände zusammen.
Das Warten, bis ein Rufknopf gedrückt wurde
und bis der Ein- und Aussteigevorgang abgeschlossen ist.


[//]: # (Welche äußeren Parameter \(Konstanten\) gibt es?)

[//]: # (Die Menschen)

## Äußere Parameter.
Die äußeren Parameter, die sich auf das Modell auswirken, 
sind die zu befördernden Personen. 
Die Fahrstühle kennen lediglich die Anzahl der Personen in ihrem Inneren 
und in welchem Stockwerk eine unbekannte Anzahl an Personen nach 
oben und / oder nach unten möchten.

[//]: # (Wie sehen die Interfaces an der Modellgrenze aus 
            \(Ein-Ausgabeparameter\)?)

[//]: # (Kapazität, Geschwindigkeit)

## Ein- und Ausgabeparameter
Um das Modell möglichst flexibel zu gestalten, wird eine Bandbreite von 
Ein- und Ausgabeparametern unterstützt. Diese können wiederum in Fahrstuhl, 
Personen und Haus unterteilt werden.

### Eingabeparametern:

Zu den Eingabeparametern der Fahrstühle gehören:

- Kapazität der Fahrstühle
- Geschwindigkeit der Etagenwechsel (in Takten)
- Dauer des Ein- und Aussteigevorgangs (in Takten)

Zu den Eingabeparametern des Hauses gehören:

- Anzahl an Etagen
- Liste von Etagen und Zeiten von Spitzenaufkommen (Bspw. Mittagspause).

Personen steuern im Gegensatz nur das maximale Tagesaufkommen zu den 
Eingaben bei.

### Ausgabeparametern:

Die Ausgabeparameter beschränken sich außerhalb des Logs lediglich auf 
die Aussagen, ob alle Personen bis zum Ende des Tages das Gebäude verlassen 
haben und welche durchschnittliche Wartezeit vorliegt.