# Implementierung nach Zombie Dice

Die erste Implementierung eines PPO-Algorithmus basiert auf die Implementierung des Professors Dr. C. Lürig für das
Spiel Zombie Dice. Dessen Netz besteht aus verschiedenen Linear Layers und verwendet die Leaky
ReLu-Aktivierungsfunktion. Als Optimierer wird der Adam Optimizer genutzt. Der Entscheidungsoutput liefert dabei die je
für ein Stockwerk für jeden Aufzug zurück. Der Input Tensor mit den Beobachtungen enthält 57 Elemente, wie zuvor in
Kapitel \ref{concept_input} beschrieben.

## Trainingsprozess nach Dice

Der Trainingsprozess des Zombie Dice Reinforcement Learners basiert auf der Kreuzentropie Methode und der Policy
Gradient Theorems. Dabei wird das Monte-Carlo Verfahren angewendet, welche ganze Episoden, in diesem Fall einen Batch
mit simulierten Tagen in Anwendung der aktuellen Policy, benötigt.

Um Daten für den Trainingsprozess zu sammeln, wurde in jeder Epoche 64 Tage mit der aktuellen Policy simuliert.
Anschließend wird der Loss berechnet und die Policy angepasst.
Die Daten eines Tages werden innerhalb der Simulation gespeichert und am Ende zurückgegeben. Da in dieser Implementation
ein Batch aus abgeschlossen simulierten Tagen besteht, ist eine schrittweise Ausführung, wie die Bibliothek SimPy
unterstützt, nicht Notwendig.

Nach dem Trainingsschritt wird das Netz 25 Mal getestet, um einen stabilen durchschnittlichen Reward zur Bewertung des
Netzes zu erhalten. Ist das aktuelle Netz besser als seine Vorgänger, so wird der aktuelle Stand zwischengespeichert.
Der Trainingsprozess ist mit dem Erreichen des Ziels, dass keine Personen sich im Bürokomplex mehr
befinden dürfen fürs Erste beendet.