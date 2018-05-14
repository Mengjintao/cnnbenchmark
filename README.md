# cnnbenchmark

We evaluate performance with VGG16, GoogleNet(Inception-V1), ResNet50, Mobilenet, Squeezenet and densenet-121 respectively, on the following 5 devices: 

|Device|Processor|\#CPUs @ Clock Speed|CPU Arch.|Memory (ms)| OS | SOC Power|
|---|---:|---:|---:|---:|---:|---|
|Samsung S8   | Snapdragon 835   | 4 @ 2.45Ghz + 4 @ 1.90GHz | Kryo       |  4GB   | Android 7.0  | ~5W   |  
|Apple iPhone 7 | A10 Fusion       | 2 @ 2.34Ghz + 2 @ 1.05GHz | Hurricane  |  2GB   | iOS 11.1     | ~5W   |
|Huawei D05 Server  |  Hi1616          | 2 * 32 @ 2.40GHz | Cortex-A72 |  256GB | Ubuntu 16.04 | >100W |
|Phytium FT1500A/16  | FTC660           | 16 @ 1.50GHz | Earth      |  64GB  | Kylin 5.0    | 35W   |
|Firefly-RK3399      | RK3399           | 2 @ 1.8Ghz + 4 @ 1.40GHz  | Cortex-A72 |  2GB   | Debian       | 6.05W |
|Raspberry Pi 3      | Broadcom BCM2837  | 4 @ 1.2Ghz               | Cortex-A53 |  1GB   | Ubuntu 16.04 | ~5W   |

To contrast, we have also tested multiple other libraries on the same devices as baseline, including `Caffe + OpenBLAS`, `Caffe2 + Eigen` and `Caffe2 + NNPACK`.

## 1. Huawei D05 Server (64-core, dual sockets)
To evaluated the scalabiltiy of state-of-art CNN inference tools, Huawei D05 Server is a domestically made many-core arm server with 64 arm A72 cores. All these 64 cores are inter-connected with a token-ring network.

#### 1.1 FeatherCNN
|Network| 1 | 2  |4  |8 | 16 | 32 | 64 | 
|---|---:|---:|---:|---:|---:|---:|---|
|[VGG16]        | 1333 | 697  | 385      | 218 |157   | 117  |  102  |
|[GoogleNet]    | 266  | 151  | 97       | 60  |  c   |  c   |  c    |
|[Resnet-50]    | 573  | 356  | 187      | 117 | 104  | 65   | 194   |
|[squeezenet]   | 153  | 98   |	58       | 44  |  c   |  c   |   c   |
|[mobilenet]    | 124  | 70   | 42	 | 36  | 34   |	52   |	76   |
|[densenet-121] | 522  | 284   | 174     | 115 |  c   |  c   |   c   |

`c` means FeatherCNN has crashed on this case. 

#### 1.2 Caffe + OpenBLAS

|Network| 1 | 2  |4  |8 | 16 | 32 | 64 | speedup | 
|---|---:|---:|---:|---:|---:|---:|---:|---|
|[VGG16]        | 3329 | 2227 |	1443 | 1108| 1137|2109  |   3721|  10.86 |
|[GoogleNet]    | 1028 | 929  | 861	 | 831 | 822 | 848  | 857 |  13.7|
|[Resnet-50]    | 728  | 490  |	347	 | 278 | 252 | 346  | 365 |  3.88|
|[squeezenet]   | 190  | 127  |	92   | 76  | 74  | 84   | 92  |      1.68|
|[mobilenet]    | 211  | 166  | 146  | 139 | 137 | 153  | 184 |     4.03 |
|[densenet-121] | 865  | 593  | 438	 | 373 | 354 | 655  | 856 |  3.08|

`speedup` is caculated with the minimum time usage of the given tool divided by the minimum time usage of FeatherCNN over all cores.

#### 1.3 Caffe2 + Eigen 

|Network| 1 | 2  |4  |8 | 16 | 32 | 64 | speedup | 
|---|---:|---:|---:|---:|---:|---:|---:|---|
|[VGG16]        | 3267 | 2173 |	1550	 | 1310|1385 | 	1323 |	1401 | 12.84 |
|[GoogleNet]    | 351  | 347  |	267      | 306 | 894 | 	2422 | 3938  |   4.45|
|[Resnet-50]    | 869  | 549  |	374	 | 262 | 149 | 	355  | 724 |     2.29|
|[squeezenet]   | 91   | 65   |	55       | 87  | 221 |  628  | 723 |     1.25|
|[mobilenet]    | 174  | 139  | 110      | 90  | 110 | 	171  |	592 |    2.65|
|[densenet-121] | x    | x    | x        |x    |x   | x   | x   |    x|

` x ` means caffe2+eigen can not successfully implement densenet-121 network. 

#### 1.4 NCNN

|Network| 1 | 2  |4  |8 | 16 | 32 | 64 |speedup | 
|---|---:|---:|---:|---:|---:|---:|---:|---|
|[VGG16]        | 1252 | 691 | 375|207 | 177 | 146 |196 | 1.43 |
|[GoogleNet]    | 320	 | 167 |102	|74	 |  67 |207	 | 290| 1.12 |
|[Resnet-50]    | 1026 |562	 |318	|180 | 112 | 150 |413 |  1.72|
|[squeezenet]   | 199	 | 115 |65	|37	 |30	 |78	 |188 | 0.68|
|[mobilenet]    | 221	 |125	 |60 |37 |44	 | 165 |199 | 1.09|
|[densenet-121] | 825	 | 536 |238 |195 |137 | 163 |1304 |  1.19|


## 2. RK3399 (2 big and 4 little cores, big.little architecture)

As ARM has a unique big.little archtecture for energy saving, to evaluate the adaptation of schduling algortihm and blocking strategies with this big.little archtecture, RK3399 is selected as an widely used embeded developing board for testing. RK3399 has 2 big cores with 1.8GHz, and 4 little cores with 1.4GHz. 

#### 2.1 FeatherCNN

|Network| 1 | 2  |1  | 2 | 4 | all  | Memory (MB) |
|---|---:|---:|---:|---:|---:|---:|---|
|[VGG16]        | 2268 | 1620 | 6122|3422 | 2269  |  1932   |   904  |
|[GoogleNet]    | 416  | 250  | 927 |524  |  333  |  294    |   168  |
|[Resnet-50]    | 857  | 517  | 1834| 1009|671    | 555     |   466  | 
|[squeezenet]   | 236  | 144  |539  | 315 |  210  |  172    |   404  |
|[mobilenet]    | 242 |  137  | 487  | 271   | 165   |  153  |   176  |
|[densenet-121] | 842  | 543  | 1854 | 1050 |  686 |  543    |   111  |

#### 2.2 Caffe + OpenBLAS

#### 2.3 Caffe2 + Eigen 

#### 2.4 NCNN
|Network| 1 | 2  |1  | 2 | 4 | all  | speedup |
|---|---:|---:|---:|---:|---:|---:|---|
|[VGG16]        | 2498 | 1976 | 5638 | 3465 |	2264 | 1627 | 1.22 |
|[GoogleNet]    | 483	 | 277	|1429  |  762	| 433	 | 465	|1.11  |
|[Resnet-50]    | 1784 | 974	| 6728 | 3489	| 1905 | 1403	|1.88  |
|[squeezenet]   | 403  |263	  |1130	 |598	  |373	 | 363	|1.82  | 
|[mobilenet]    | 335	 |192	  |1250	 |663	  | 378	 |330	  |2.41  |  
|[densenet-121] | 1323 |761	  | 5360 |2819	|1574	 | 1612	|1.4   |


## 3. Raspberry Pi 3 (4 A53 cores)
#### 3.1 FeatherCNN

|Network| 1 | 2  | 4 | 
|---|---:|---:|---|
|[VGG16]        | -    | -    |  -       |
|[GoogleNet]    | 1058 | 642  | 809      |
|[Resnet-50]    | 2107 | 1255 | 1540     |
|[squeezenet]   | 638  | 399  | 501      |
|[mobilenet]    | 451  |  275 | 206	 | 
|[densenet-121] | 630   | 396 | 459      |

#### 3.2 Caffe + OpenBLAS

#### 3.3 Caffe2 + Eigen 

#### 3.4 NCNN

|Network| 1 | 2  | 4 |  speedup | 
|---|---:|---:|---:|---|
|[VGG16]        | -    | -    |  -       |   -   |
|[GoogleNet]    | 1896 | 1018	| 1130	   |  1.58  | 
|[Resnet-50]    | 8386 |4392	|3987	     |3.17    |
|[squeezenet]   | 1268 |694	  |760	     |1.74    |
|[mobilenet]    | 1758 |951	  |570	     |2.7     |
|[densenet-121] | 1268 |694	  |760	     |1.74    |
	




#### TX2 (2 big and 4 little cores, big.little architecture)

|Network| 1 | 2  |1  | 2 | 4 | all  |
|---|---:|---:|---:|---:|---:|---|
|[VGG16]        | 1325 | 706  | 2540     |1507 | 1226  |  844  |
|[GoogleNet]    | 274  | 146 | 366       |206  |  127  |  105  |
|[Resnet-50]    | 480  | 266  | 759     | 417  |261    | 215   | 
|[squeezenet]   | 88   | 115  |73       | 61   | 204   |  153  |
|[mobilenet]    | 156 |  87   | 211      | 116 | 68    |  56   |
|[densenet-121] | -    | -    | -         | - |   -    |  -   |


#### Caffe2 + NNPACK 
