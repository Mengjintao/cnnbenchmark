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

## Huawei D05 Server (64-core, dual sockets)
#### 1.1 FeatherCNN
|Network| 1 | 2  |4  |8 | 16 | 32 | 64 | 
|---|---:|---:|---:|---:|---:|---:|---|
|[VGG16]        | 1333 | 697  | 385      | 218 |157   | 117  |  102  |
|[GoogleNet]    | 266  | 151  | 97       | 60  |  -   |  -   |  -    |
|[Resnet-50]    | 573  | 356  | 187      | 117 | 104  | 65   | 194   |
|[squeezenet]   | 153  | 98   |	58       | 44  |  -   |  -   |   -   |
|[mobilenet]    | 124  | 70   | 42	     | 36  | 34   |	52   |	76   |
|[densenet-121] | 522  | 284   | 174     | 115 |  -   |  -   |   -   |

#### 1.2 Caffe + OpenBLAS

|Network| 1 | 2  |4  |8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|---:|---|
|[VGG16]        | 23626	| 15127 |	8662 | 	6206 |	4776 |	4393 | 	4900 |
|[GoogleNet] | 1028 | 929  | 861	 | 831 | 822 | 848  | 857 |
|[Resnet-50]    | 728  | 490  |	347	 | 278 | 252 | 346  | 365 |
|[squeezenet]   | 190  | 127  |	92   | 76  | 74  | 84   | 92  |
|[mobilenet]    | 211  | 166  | 146  | 139 | 137 | 153  | 184 |
|[densenet-121] | 865  | 593  | 438	 | 373 | 354 | 655  | 856 |

#### 1.3 Caffe2 + Eigen 

|Network| 1 | 2  |4  |8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|---:|---|
|[VGG16]        | 3267 | 2173 |	1550	 | 1310|1385 | 	1323 |	1401 |
|[GoogleNet]    | 351  | 347  |	267      | 306 | 894 | 	2422 | 3938  |
|[Resnet-50]    | 869  | 549  |	374	 | 262 | 149 | 	355  | 724 |
|[squeezenet]   | 91   | 65   |	55       | 87  | 221 |  628  | 723 |
|[mobilenet]    | 174  | 139  | 110      | 90  | 110 | 	171  |	592 |
|[densenet-121] | -  | -  | -	 |- | - | -  | - |




#### RK3399 (2 big and 4 little cores, big.little architecture)

|Network| 1 | 2  |1  | 2 | 4 | all  | Memory (MB) |
|---|---:|---:|---:|---:|---:|---:|---|
|[VGG16]        | 2268 | 1620 | 6122|3422 | 2269  |  1932   |   904  |
|[GoogleNet]    | 416  | 250  | 927 |524  |  333  |  294    |   168  |
|[Resnet-50]    | 857  | 517  | 1834| 1009|671    | 555     |   466  | 
|[squeezenet]   | 236  | 144  |539  | 315 |  210  |  172    |   404  |
|[mobilenet]    | 242 |  137  | 487  | 271   | 165   |  153  |   176  |
|[densenet-121] | 842  | 543  | 1854 | 1050 |  686 |  543    |   111  |


#### Raspberry Pi 3 (4 cores)

|Network| 1 | 2  | 4 | 
|---|---:|---:|---|
|[VGG16]        | -    | -    |  -       |
|[GoogleNet]    | 1058 | 642  | 809      |
|[Resnet-50]    | 2107 | 1255 | 1540     |
|[squeezenet]   | 630  | 396  | 459      |
|[mobilenet]    | 451  |  275 | 206	     | 
|[densenet-121] | -    | -    | -        |


#### TX2 (2 big and 4 little cores, big.little architecture)

|Network| 1 | 2  |1  | 2 | 4 | all  |
|---|---:|---:|---:|---:|---:|---|
|[VGG16]        | 1325 | 706  | 2540     |1507 | 1226  |  844  |
|[GoogleNet]    | 274  | 146 | 366       |206  |  127  |  105  |
|[Resnet-50]    | 480  | 266  | 759     | 417  |261    | 215   | 
|[squeezenet]   | 88   | 115  |73       | 61   | 204   |  153  |
|[mobilenet]    | 156 |  87   | 211      | 116 | 68    |  56   |
|[densenet-121] | -    | -    | -         | - |   -    |  -   |




#### Caffe + OpenBLAS

|Network| 1 | 2  |4  |8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|---:|---|
|[VGG16]        | 23626	| 15127 |	8662 | 	6206 |	4776 |	4393 | 	4900 |
|[GoogleNet] | 1028 | 929  | 861	 | 831 | 822 | 848  | 857 |
|[Resnet-50]    | 728  | 490  |	347	 | 278 | 252 | 346  | 365 |
|[squeezenet]   | 190  | 127  |	92   | 76  | 74  | 84   | 92  |
|[mobilenet]    | 211  | 166  | 146  | 139 | 137 | 153  | 184 |
|[densenet-121] | 865  | 593  | 438	 | 373 | 354 | 655  | 856 |

#### Caffe2 + Eigen 

|Network| 1 | 2  |4  |8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|---:|---|
|[VGG16]        | 3267 | 2173 |	1550	 | 1310|1385 | 	1323 |	1401 |
|[GoogleNet]    | 351  | 347  |	267      | 306 | 894 | 	2422 | 3938  |
|[Resnet-50]    | 869  | 549  |	374	 | 262 | 149 | 	355  | 724 |
|[squeezenet]   | 91   | 65   |	55       | 87  | 221 |  628  | 723 |
|[mobilenet]    | 174  | 139  | 110      | 90  | 110 | 	171  |	592 |
|[densenet-121] | -  | -  | -	 |- | - | -  | - |

#### Caffe2 + NNPACK 
