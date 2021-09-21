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
 
size_zeros = len(zero_tensors)
size_fives = len(five_tensors)
 
zeros_meter = np.zeros(size_zeros)
fives_meter = np.zeros(size_fives)
 
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
 
#Result
 
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

