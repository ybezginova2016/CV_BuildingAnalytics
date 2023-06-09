# Computer Vision: Аналитика зданий

## Описание проекта
* **Заказчик - строительная компания, которая занимается возведением многоквартирных домов.**
* **Заказчику необходимо подсчитать сколько в уже построенном районе находится окон в жилых зданиях, а также количество этажей в здании и количество колонн (вертикальных рядов окон).**
* **Также заказчик хотел бы выделять на фото фасад "главного" здания, если на изображении несколько зданий.**

## Задача проекта

Обучить собственный алгоритм (Pytorch, Tensorflow) сегментации фасада главного здания, а также алгоритм определения числа окон, числа этажей и числа колонн (вертикальных рядов окон). То есть решить задачу сегментации и произвести подсчет числа окон, числа этажей и числа колонн. 

Задача сегментации — это определение класса каждого пикселя в изображения. В процессе модель создаёт маску сегментации. Основная архитектура для задачи сегментации — U-Net. В задаче сегментации экземпляров (англ. instance segmentation) каждому пикселю присваивается определённый экземпляр класса. 

*В компьютерном зрении, сегментация — это процесс разделения цифрового изображения на несколько сегментов (суперпиксели). Цель сегментации заключается в упрощении и/или изменении представления изображения, чтобы его было проще и легче анализировать.*

## Методология

### Алгоритм сегментации объектов

**Классическая модель для семантической сегментации — U-Net.**

Как и многие другие модели компьютерного зрения, U-Net — это нейронная сеть. Она была разработана, чтобы находить патологии на медицинских снимках. U-Net «смотрит» на картинку в разных масштабах и определяет, к какому классу относится каждый из пикселей.

### Функция потерь в задаче сегментации
Функция потерь в задаче семантической сегментации в простом случае — кросс-энтропия на пиксельном уровне. Для сегментации экземпляров используются более сложные функции потерь.

### 1) Изучениe данных
Изучим один из датасетов: Paris Art Deco Facades Dataset, TMBuD Dataset или CMP Facade Database. Определим, какие изображения содержат информацию о жилых зданиях и используйте их для обучения и тестирования модели.

### 2) Подготовка данных и разметка
Для разметки данных вам потребуется создать размеченные образцы с информацией о количестве окон, этажей, колонн и маске сегментации фасада главного здания. Вы можете использовать разметчики изображений, такие как labelbox или RectLabel для ручной разметки данных.

Разметка данных будет сделана в соответствии с приложенной к датасету документацией:
* 1 background 1
* 2 facade 2
* 3 window 10
* 4 door 5
* 5 cornice 11
* 6 sill 3
* 7 balcony 4
* 8 blind 6
* 9 deco 8
* 10 molding 7
* 11 pillar 12
* 12 shop 9

Опишем на русском языке разметку, чтобы лучше понимать, как рассчитать количество окон, этажей и колонн, необходимые клиенту.

* *фон* - относится к части изображения, которая не считается частью фасада здания или какого-либо его архитектурного элемента.
* *фасад* - относится к основной фронтальной стороне здания, которая анализируется.
* *окно* - относится к любому видимому окну в фасаде здания.
* *дверь* - относится к любой видимой двери в фасаде здания.
* *карниз* - относится к декоративной форме или горизонтальной выступающей части, которая завершает верхнюю часть здания или стены.
* *подоконник* - относится к нижней горизонтальной части рамы окна, которая лежит на подоконнике окна.
* *балкон* - относится к платформе, выступающей из стены здания и огражденной.
* *жалюзи* - относится к устройству, которое используется для закрытия окна или защиты от солнца.
* *декор* - относится к любому другому декоративному элементу на фасаде здания.
* *молдинг* - относится к декоративной форме, которая обрамляет окно или дверь.
* *колонна* - относится к вертикальной несущей части архитектурного элемента здания, имеющей круглую или многоугольную форму.
* *магазин* - относится к любому магазину, расположенному на первом этаже здания.

### 3) Преобработка данных
* Процедура предполагает как минимум два этапа: изменение размера изображений до подходящего для сети и нормализация изображений. Именно на этом этапе происходят аугментации — повороты изображений, изменения яркости, насыщености, случайный кроп или добавление шума. 
* Нормализуем изображения и маски, разделим данные на обучающую, валидационную и тестовую выборки.

### 4) Обучение модели
Для задачи сегментации можно использовать различные модели и архитектуры. Некоторые из популярных выборов:

* Fully Convolutional Networks (FCN): это архитектура, которая использует только сверточные слои для обработки входных данных и предсказания классов.
* U-Net: это архитектура, которая включает в себя контрастные мосты, которые позволяют сохранять информацию о мелких деталях входных данных во время декодирования.
* Mask R-CNN: это архитектура, которая использует комбинацию сверточных слоев и слоев полносвязных сетей для сегментации и детектирования объектов.

### 5) Метрики качества сегментации
В задачах семантической сегментации каждому пикселю присваивается определённый класс, что позволяет посчитать попиксельную долю правильных ответов (англ. Pixel Accuracy). Она равна доле пикселей, для которой модель правильно присвоила класс. 

Для семантической сегментации также подходит метрика IoU. Вместо прямоугольников учитываются фигуры, ограниченные контурами.
Помимо IoU можно использовать коэффициент Сёренсена-Дайса, или коэффициент Дайса, (англ. Dice coefficient) — отношение удвоенной площади пересечения к сумме площадей предсказания и реального объекта.

### 6) Применение модели
Примените обученную модель к новым изображениям жилых зданий и получите предсказания для количества окон, этажей и колонн. Также вы можете визуализировать результаты сегментации фасада главного здания.

## Данные (на выбор)

* **[Paris Art Deco Facades Dataset](https://github.com/raghudeep/ParisArtDecoFacadesDataset/)**
* **[TMBuD Dataset, Feb'2020](https://github.com/CipiOrhei/TMBuD)** (парсинг базы)
* **[CMP Facade Database (extended dataset)](https://cmp.felk.cvut.cz/~tylecr1/facade/)**

## Итоги:
- основной ноутбук: https://github.com/ybezginova2016/CV_BuildingAnalytics/blob/main/model.ipynb

- analysis.ipynb - файл с исходной аналитикой данных, подсчетом предметов на масках. Он используется в паре с labels.csv, где выведены векторизованные метки.

- DataGen.py - написан класс для формирования и предобработки данных перед обучением модели.

- model.ipynb - основной ноутбук с моделями, которые были обучены. Здесь же сохраняется модель в формате .h5 и приводится вывод, какая модель отработала лучше всего и может быть использована на живых данных.

- load_predict.py - файл с предсказаниями на живых данных, может быть использован для деплоя модели.

## Форма для заполнения
https://docs.google.com/forms/d/e/1FAIpQLSfkavbuwPY20eQVS_dx7daQBGoBnDlGRtbfWzMKdi_tOFUkYg/viewform 

