# AIDAO-Final
### В этом репозитории представлено решение моей команды на этой олимпиаде.
AIDAO - Artificial intelligence and data analysis Olympiad - Международная олимпиада по искусственному интеллекту и анализу данных 
от Яндекс.Образования и ВШЭ проводилась во второй половине 2024 года. Финал олимпиады проходил в Москве 1-2 декабря. 

Основная задача была предоставлена сервисом 
Яндекс.Такси: водителям переиодически необходимо отправлять в приложение 4 фотографии своего автомобиля с разных ракурсов и участникам
было предложено разарботать модель, которая проверяла бы, что на фотографиях представлены все 4 стороны автомобиля и по всем сторонам 
сделать вывод: годна машина к работе или нет. Задача бинарной классификации, метрика - ROC AUC.

### Задача была примечательна тем, что для инференса использовалось CPU, а не GPU.
Железо для инференса модели: Intel(R) Xeon(R) Gold CPU 6338 @ 2.00GHz (single core virtual machine), 1.5 GB RAM.
Не более 30 минут на сборку контейнера и инференс.
