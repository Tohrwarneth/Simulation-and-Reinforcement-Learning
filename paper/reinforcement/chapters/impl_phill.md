# Implementierung nach Neuralnet \label{impl_phill}

2021 veröffentlichte der YouTube-Kanal [MachineLearningwithPhil](https://www.youtube.com/@MachineLearningwithPhil/) ein
Video zur Implementierung eines einfachen Reinforcement Learners mit Proximal Policy Optimization. Als Simulation wurde
auf die Gym Bibliothek zurückgegriffen, welche kleine Beispielsimulationen bereithält, auf denen der eigene
Reinforcement Learner trainieren kann.

## Trainingsprozess nach Neuralnet

Der Reinforcement Learner von Neuralnet basiert auf die in Kapitel \ref{formel_OpenAI} gezeigte Formel von OpenAI.

Im Gegensatz zur Implementierung nach Zombie-Dice i, voran gegangenen Kapitel \ref{train_dice}, wird nicht erst mehrere
Episoden mit der aktuellen Policy simuliert und anschließend ausgewertet, sondern setzt den Lernprozess noch während der
Simulation ein. Im Original-Code wird alle 20 Takte der Trainingsprozess gestartet und eine neue Policy bestimmt.

Diese Trainingsrate ist jedoch im Falle der Fahrstuhlsimulation zu hoch gewählt, da in den ersten 700 Takten keine
Personen den Bürokomplex betreten und somit jegliche Entscheidungen des Reinforcement Deciders keine Auswirkung auf die
Simulation zeigt. Daher wurde sich für eine Trainingsrate von 8 Iterationen pro simulierten Tag, etwa alle 3 Stunden,
entschieden. Zwar zeigen so die ersten zwei Iterationen keine Änderungen durch die Entscheidungen, jedoch die restlichen
6 Iterationen sind die wichtigsten Zeiten, in denen die Befehle der Fahrstühle eine hohe Auswirkung zeigen.
In einer weiteren Variante könnte man die ersten zwei Iterationen überspringen, und somit diesen Effekt verringern.
Dies wurde erst in der Auswertung des Trainings erkannt und nicht mehr rechtzeitig für weitere Trainingsversuche
notiert.

Das Netz dieser Implementierung ähnelt dem aus der Implementierung nach Zombie-Dice. Jedoch werden hier die zwei Köpfe
als getrennte Netze behandelt.