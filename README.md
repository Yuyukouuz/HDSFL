# HDSFL
Official Pytorch implementation of "Federated Learning for Spiking Neural Networks by Hint-layer Knowledge Distillation"

## Code Description
compress.py：Spike tensor compression.

confusion_matrix.py：Draw confusion matrix.

dataset_config.py：Dataset configuration code, used to configure public and private datasets.

dvs_loader.py：Load dvs dataset.

flsnn_client.py：The core code of the client, which includes all the operations and order that the client can perform, with the main loop being the client_start().

flsnn_server.py：The core code of the server, which includes server operations and coordinating client operations, with the main loop being the server_start().

global_configs.py：Global parameter configuration, the meaning of configuration parameters has been annotated.

main_client.py：Client entrance.

main_server.py：Server entrance.

newVGG.py：GRSNN model.

newVGG_layers.py：GRSNN network layer settings.

read_record.py：Read saved results.

SCNN.py：DTSNN model.

-SVGG.py：VSNN model.

tools.py：Specific implementation of subroutines, including model saving, loss function, sample acquisition, data structure transformation, general network training, composite distillation training, accuracy testing, output spike tensor, and acquisition of confusion matrix.

## Training steps
1. Configure parameters in global_configs.py and save.

2. Run main_server.py.

3. Run main_client.py. Each client needs to run once in a new window, and the GPU serial number used by the client can be configured before running

4. Wait for the output result. The output result will be saved in running_save directory.

## Read experimental results：
Run read_record.py --open filename --print True
