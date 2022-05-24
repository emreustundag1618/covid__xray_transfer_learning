# Covid-19 Classification from Chest X-Ray Images with Deep Transfer Learning Methods

## Emre Üstündağ, M.Sc Thesis

### Graduate School of Natural and Applied Sciences - Department of Statistics

[Summary]

| **Sınıflandırma tipi** | **ESA Modeli** | **ESA** | **ESA + DVM** |
| --- | --- | --- | --- |
| İkili(Covid-19 pozitif ya da negatif) | ResNet50V2InceptionV3XCeptionCheXNet | 0.7850.7960.800 **0.807** | 0.7180.7990.7220.794 |
| Çoklu(Tipik, atipik, belirsiz, normal) | ResNet50V2InceptionV3XCeptionCheXNet | 0.7470.7500.744 **0.771** | 0.6750.7260.6990.716 |
| İkili(Tipik görünüm ya da negatif) | ResNet50V2InceptionV3XCeptionCheXNet | 0.8080.8350.8330.846 | 0.779 **0.856** 0.8400.832 |
| İkili(Atipik görünüm ya da negatif) | ResNet50V2InceptionV3XCeptionCheXNet | 0.5000.5800.5310.645 | 0.5000.6360.582 **0.683** |
| İkili(Belirsiz görünüm ya da negatif) | ResNet50V2InceptionV3XCeptionCheXNet | 0.6240.6860.5340.698 | 0.6200.6600.689 **0.752** |

