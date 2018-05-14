# cnnbenchmark

We evaluate performance with VGG16, GoogleNet(Inception-V1), ResNet50, Mobilenet, Squeezenet and densenet-121 respectively, on the following 5 devices: 

|Network|Layers|Top-1 error|Top-5 error|Speed (ms)|Citation|
|---|---:|---:|---:|---:|---|
|[AlexNet](#alexnet)|8|42.90|19.80|14.56|[[1]](#alexnet-paper)|
|[Inception-V1](#inception-v1)|22|-|10.07|39.14|[[2]](#inception-v1-paper)|
|[VGG-16](#vgg-16)      |16|27.00|8.80|128.62|[[3]](#vgg-paper)|
|[VGG-19](#vgg-19)      |19|27.30|9.00|147.32|[[3]](#vgg-paper)|
|[ResNet-18](#resnet-18)|18|30.43|10.76|31.54|[[4]](#resnet-cvpr)|
|[ResNet-34](#resnet-34)|34|26.73|8.74|51.59|[[4]](#resnet-cvpr)|
|[ResNet-50](#resnet-50)|50|24.01|7.02|103.58|[[4]](#resnet-cvpr)|
|[ResNet-101](#resnet-101)|101|22.44|6.21|156.44|[[4]](#resnet-cvpr)|
|[ResNet-152](#resnet-152)|152|22.16|6.16|217.91|[[4]](#resnet-cvpr)|
|[ResNet-200](#resnet-200)|200|21.66|5.79|296.51|[[5]](#resnet-eccv)|


|Network|Layers|Top-1 error|Top-5 error|Speed (ms)|Citation|
|---|---:|---:|---:|---:|---|
|AlexNet|8|42.90|19.80|14.56|1|

|Network|1|2|4|8|16|32|64| 
|---|---:|---:|---:|---:|---|
|VGG16|1333|697|385|218|157|117|102|
|GoogleNet|266|151|97|60|-|-|-|
|Resnet-50|573|356|187|117|104|65|194|
|squeezenet|153|98|58|44|-|-|-|
|mobilenet|124|70|42|36|34|52|76|
|densenet-121|522|284|174|115|-|-|-|



|Device|Processor|\#CPUs @ Clock Speed|CPU Arch.|Memory (ms)| OS | SOC Power|
|---|---:|---:|---:|---:|---|
|Samsung Galaxy S8   | Snapdragon 835   | 4 @ 2.45Ghz + 4 @ 1.90GHz | Kryo       |  4GB   | Android 7.0  | ~5W   |  
|Apple iPhone 7 plus | A10 Fusion       | 2 @ 2.34Ghz + 2 @ 1.05GHz | Hurricane  |  2GB   | iOS 11.1     | ~5W   |
|Huawei D05 Server   |  Hi1616          | 2 * 32 @ 2.40GHz | Cortex-A72 |  256GB | Ubuntu 16.04 | >100W |
|Phytium FT1500A/16  | FTC660           | 16 @ 1.50GHz | Earth      |  64GB  | Kylin 5.0    | 35W   |
|Firefly-RK3399      | RK3399           | 2 @ 1.8Ghz + 4 @ 1.40GHz  | Cortex-A72 |  2GB   | Debian       | 6.05W |
|Raspberry Pi 3      | ARM A53          | 4 @ 1.2Ghz                | Cortex-A53 |  1GB   | Ubuntu 16.04 | -     |


To contrast, we have also tested multiple other libraries on the same devices as baseline, including `Caffe + OpenBLAS`, `Caffe2 + Eigen` and `Caffe2 + NNPACK`.

## FeatherCNN
#### Huawei D05 Server (64-core, dual sockets)

|Network| 1 | 2  |4  |8 | 16 | 32 | 64 | 
|---|---:|---:|---:|---:|---|
|[VGG16]        | 1333 | 697  | 385      | 218 |157   | 117  |  102  |
|[GoogleNet]    | 266  | 151  | 97       | 60  |  -   |  -   |  -    |
|[Resnet-50]    | 573  | 356  | 187      | 117 | 104  | 65   | 194   |
|[squeezenet]   | 153  | 98   |	58       | 44  |  -   |  -   |   -   |
|[mobilenet]    | 124  | 70   | 42	 | 36  | 34   |	52   |	76   |
|[densenet-121] | 522  | 284   | 174     | 115 |  -   |  -   |   -   |

#### RK3399 (2 big and 4 little cores, big.little architecture)

|Network| 1 | 2  |1  | 2 | 4 | all  |
|---|---:|---:|---:|---:|---|
|[VGG16]        | 2268 | 1620 | 6122     |3422 | 2269  |  1932   |
|[GoogleNet]    | 416  | 250  | 927      |524  |  333  |  294    |
|[Resnet-50]    | 857  | 517  | 1834     | 1009|671    | 555     | 
|[squeezenet]   | 236  | 144  |539       | 315 |  210  |  172    |
|[mobilenet]    | 242 |  137   | 487	   | 271  | 165  |  153    |
|[densenet-121] | 842  | 543  | 1854     | 1050 |  686 |  543    |


#### Raspberry Pi 3 (4 cores)

|Network| 1 | 2  | 4 | Memory (MB) |
|---|---:|---:|---:|---:|---|
|[VGG16]        | -    | -    |  -       |   -  |
|[GoogleNet]    | 1058 | 642  | 809      |   -  |
|[Resnet-50]    | 2107 | 1255 | 1540     |   -  | 
|[squeezenet]   | 630  | 396  | 459      |   -  |
|[mobilenet]    | 451  |  275 | 206	     |   -  |
|[densenet-121] | -    | -    | -        |   -  |


#### TX2 (2 big and 4 little cores, big.little architecture)

|Network| 1 | 2  |1  | 2 | 4 | all  |
|---|---:|---:|---:|---:|---|
|[VGG16]        | 1325 | 706  | 2540     |1507 | 1226  |  844  |
|[GoogleNet]    | 274  | 146 | 366       |206  |  127  |  105  |
|[Resnet-50]    | 480  | 266  | 759     | 417  |261    | 215   | 
|[squeezenet]   | 88   | 115  |73       | 61   | 204   |  153  |
|[mobilenet]    | 156 |  87   | 211      | 116 | 68    |  56   |
|[densenet-121] | -    | -    | -         | - |   -    |  -   |




#### Caffe + OpenBLAS

|Network| 1 | 2  |4  |8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---|
|[VGG16]        | 23626	| 15127 |	8662 | 	6206 |	4776 |	4393 | 	4900 |
|[GoogleNet] | 1028 | 929  | 861	 | 831 | 822 | 848  | 857 |
|[Resnet-50]    | 728  | 490  |	347	 | 278 | 252 | 346  | 365 |
|[squeezenet]   | 190  | 127  |	92   | 76  | 74  | 84   | 92  |
|[mobilenet]    | 211  | 166  | 146  | 139 | 137 | 153  | 184 |
|[densenet-121] | 865  | 593  | 438	 | 373 | 354 | 655  | 856 |

#### Caffe2 + Eigen 

|Network| 1 | 2  |4  |8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---|
|[VGG16]        | 3267 | 2173 |	1550	 | 1310|1385 | 	1323 |	1401 |
|[GoogleNet]    | 351  | 347  |	267      | 306 | 894 | 	2422 | 3938  |
|[Resnet-50]    | 869  | 549  |	374	 | 262 | 149 | 	355  | 724 |
|[squeezenet]   | 91   | 65   |	55       | 87  | 221 |  628  | 723 |
|[mobilenet]    | 174  | 139  | 110      | 90  | 110 | 	171  |	592 |
|[densenet-121] | -  | -  | -	 |- | - | -  | - |

#### Caffe2 + NNPACK 
