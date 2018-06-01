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
|Huawei Mate10       |  Kirin 970        | 4A73 @ 2.34Ghz + 4 A53@ 1.05GHz      | Cortex-A73/A53 |  4GB  | Ubuntu 16.04 | ~5W  |


To contrast, we have also tested multiple other libraries on the same devices as baseline, including `Caffe + OpenBLAS`, `Caffe2 + Eigen` and `Caffe2 + NNPACK`.

## 1. Huawei D05 Server (64-core, dual sockets)
To evaluated the scalabiltiy of state-of-art CNN inference tools, Huawei D05 Server is a domestically made many-core arm server with 64 arm A72 cores. All these 64 cores are inter-connected with a token-ring network.

#### 1.1 FeatherCNN-F(2x2,3x3)
|Network| 1 | 2  |4  |8 | 16 | 32 | 64 | 
|---|---:|---:|---:|---:|---:|---:|---|
|[VGG16]        | 1333 | 697  | 385      | 218 |157   | 117  |  102  |
|[GoogleNet]    | 333	| 210 | 154	 |125  |126   |151   | 230   |
|[Resnet-50]    | 573  | 356  | 187      | 117 | 104  | 65   | 194   |
|[squeezenet]   | 149  |79    |	44       |28	|29   |35    | 67    |
|[mobilenet]    | 124  | 70   | 42	 | 36  | 34   |	52   |	76   |
|[densenet-121] | 517  |273   | 156      |98   | 113  | 160  | 331   |

#### 1.1 FeatherCNN-F(6x6,3x3)
|Network| 1 | 2  |4  |8 | 16 | 32 | 64 | 
|---|---:|---:|---:|---:|---:|---:|---|
|[VGG16]        | - | -  | -      | - |-   | -  |  -  |
|[GoogleNet]    | - | -  | -      | - |-   | -  |  -  |
|[Resnet-50]    | - | -  | -      | - |-   | -  |  -  |
|[squeezenet]   | - | -  | -      | - |-   | -  |  -  |
|[mobilenet]    | - | -  | -      | - |-   | -  |  -  |
|[densenet-121] | - | -  | -      | - |-   | -  |  -  |

## 2. RK3399 (2 big and 4 little cores, big.little architecture)

As ARM has a unique big.little archtecture for energy saving, to evaluate the adaptation of schduling algortihm and blocking strategies with this big.little archtecture, RK3399 is selected as an widely used embeded developing board for testing. RK3399 has 2 big cores with 1.8GHz, and 4 little cores with 1.4GHz. 

#### 2.1 FeatherCNN-F(2x2,3x3)

|Network| 1 | 2  |1  | 2 | 4 | all  | Memory (MB) |
|---|---:|---:|---:|---:|---:|---:|---|
|[VGG16]        | 2268 | 1620 | 6122|3422 | 2269  |  1932   |   904  |
|[GoogleNet]    | 416  | 250  | 927 |524  |  333  |  294    |   168  |
|[Resnet-50]    | 857  | 517  | 1834| 1009|671    | 555     |   466  | 
|[squeezenet]   | 236  | 144  |539  | 315 |  210  |  172    |   404  |
|[mobilenet]    | 242 |  137  | 487  | 271   | 165   |  153  |   176  |
|[densenet-121] | 842  | 543  | 1854 | 1050 |  686 |  543    |   111  |


## 3. Raspberry Pi 3 (4 A53 cores)
#### 3.1 FeatherCNN-F(2x2,3x3)

|Network| 1 | 2  | 4 | 
|---|---:|---:|---|
|[VGG16]        | -    | -    |  -       |
|[GoogleNet]    | 1058 | 642  | 809      |
|[Resnet-50]    | 2107 | 1255 | 1540     |
|[squeezenet]   | 638  | 399  | 501      |
|[mobilenet]    | 451  |  275 | 206	 | 
|[densenet-121] | 630   | 396 | 459      |


## Apple iPhone 7 plus and Samsung S8
   @bug1987 can you help us collect the data for iPhone 7 plus and Samsung S8 on NCNN, Caffe, and Caffe2




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
