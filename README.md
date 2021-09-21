# Classificador-de-digitos
Este projeto é um classificador de dígitos que diferencia os algarismos 0 e 5 do dataset MNIST.

Para isso, foi utilizada a seguinte lógica: 

Considere os algarismos 0 e 5 representados no plano, ambos com traços de espessura infinitesimal. Se fizermos uma reta vertical no centro de cada dígito com o mesmo tipo de traço, poderemos enxergar que existem dois pontos de interseção entre a reta e o algarismo 0 e três interseções entre a reta e o algarismo 5, como mostra a figura abaixo:
![Figura 1](https://user-images.githubusercontent.com/66181468/134167355-424debca-7fe8-469e-b617-0e0a7fc98271.png)

Podemos adaptar este raciocínio para lógica de programação e testar os resultados obtidos.

#### Detalhamento do código

O algoritmo foi escrito em Python, com uso das bibliotecas numpy e fastbook. Além disso, foi utilizado como base a teoria e código exemplo do notebook da FastAI disponibilizado pela empresa.

Primeiramente foram feitas as importações, download do MNIST e armazenamento dos dígitos de interesse (0 e 5) em tensors.
```
import numpy as np
import fastbook
fastbook.setup_book()
 
from fastai.vision.all import *
from fastbook import *
 
path = untar_data(URLs.MNIST)
 
zeros = (path/'training'/'0').ls().sorted()
fives = (path/'training'/'5').ls().sorted()
 
zero_tensors = [tensor(Image.open(i)) for i in zeros]
five_tensors = [tensor(Image.open(i)) for i in fives]
```

Depois, foram criadas variáveis para armazenar a quantidade de imagens de zeros e cincos do banco de dados. Também foram criados vetores contadores para cada imagem, inicializados com 0 em todas as posições.

```
size_zeros = len(zero_tensors)
size_fives = len(five_tensors)
 
zeros_meter = np.zeros(size_zeros)
fives_meter = np.zeros(size_fives)
```

Agora é o momento onde a lógica descrita no início deste documento é implementada. 
Lá tínhamos um traço infinitesimal, o que era ideal para encontrarmos pontos de interseção.

Aqui, queremos o que pode ser interpretado como um pixel de interseção. Porém não temos uma imagem do dígito formada por um “traço” de apenas um pixel, e também não iremos desenhar uma reta na imagem.

O que faremos no lugar da reta, será percorrer uma única coluna (14) da matriz da imagem procurando por valores diferentes de 0. Entretanto, como temos mais de um pixel formando o “traço” do dígito, é preciso adotar uma estratégia para que apenas um pixel seja contabilizado, representando o ponto de interseção.

Esta estratégia consiste em considerar valores diferentes de zero apenas se o pixel de cima for zero (ou branco).

No final, devemos ter o número de interseções, representado pelo contador sample_meter neste caso, que vai nos dizer se o dígito é um 0 ou 5. Se o contador armazenar um valor menor que 3 ele será classificado como 0, caso contrário como 5.

```
#Sample:
 
sample_0 = zeros[1234]
sample_0 = Image.open(sample_0)
sample_meter = 0
 
i=0
while i < 28:
   if array(sample_0)[i,14] != 0:
       if array(sample_0)[i-1,14] == 0:
           sample_meter += 1
   i += 1
 
if sample_meter < 3:
   print("ZERO") #    <- in this case
else:
   print("FIVE")
   ```
  
Para classificar outras imagens basta substituir o valor de sample_0.

Vimos que neste caso o classificador foi eficaz. Vamos verificar se será sempre assim.

#### Resultados

```
Result
 
j = 0
meter = 0
while j < size_zeros:
   k = 0
   while k < 28:
       if array (zero_tensors[j])[k,14] != 0:
           if array (zero_tensors[j])[k-1,14] == 0:
               zeros_meter[j] +=1
       k += 1
   if zeros_meter[j] < 3:
       meter += 1
   j += 1
 
print(meter/size_zeros)
# OUT: 0.9680904946817491

j = 0
meter = 0
while j < size_fives:
   k=0
   while k < 28:
       if array (five_tensors[j])[k,14] != 0:
           if array (five_tensors[j])[k-1,14] == 0:
               fives_meter[j] +=1
       k += 1
   if fives_meter[j] >= 3:
       meter += 1
   j += 1
 
print(meter/size_fives)
# OUT: 0.6275594908688434
```

De acordo com o resultado obtido, é possível perceber que 96,81% das imagens do banco de dados que contém o dígito zero serão classificadas corretamente pelo algoritmo criado!
Também temos que 62,73% das imagens do MNIST que contém o dígito cinco serão classificadas corretamente.

#### Conclusão

Foi desenvolvido um classificador muito simples, com 79,78% de eficácia e otimizado, já que não é necessário o carregamento do banco de dados a cada vez que ele é utilizado. Além disso, apenas uma coluna da imagem é percorrida, o que economiza muito tempo de processamento. 
