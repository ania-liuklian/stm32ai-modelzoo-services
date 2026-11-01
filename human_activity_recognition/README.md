# Human Activity Recognition STM32 Model Zoo


## Directory components:
* [config_file_examples](./config_file_examples/) includes the sample minimalistic configuration files to run different operation modes, such as [training](./config_file_examples/training_config.yaml), [deployment](./config_file_examples/deployment_config.yaml), [evaluation](./config_file_examples/evaluation_config.yaml) and [benchmarking](./config_file_examples/benchmarking_config.yaml). These files can be used directly to launch one of the operation modes with little to no editions.
* [datasets](./datasets/) placeholder for the human activity recognition datasets.
* [docs](./docs/) contains all readmes and documentation specific to the human activity recognition use case.
* [tf](./tf/) contains all the source code to build, train, evaluate, benchmark, quantize and deploy your human_activity_recognition model on your STM32 target.

## Quick & easy examples:
The `operation_mode` top-level attribute specifies the operations or the service you want to execute. This may be single operation or a set of chained operations.

## Train and evaluate on CPU (WSL/Ubuntu)

Run from `human_activity_recognition` in WSL/Ubuntu (Python 3.12):

```bash
python3.12 -m venv st_zoo
source st_zoo/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-cpu.txt
python stm32ai_main.py --config-path ./config_file_examples/ --config-name training_config_mobility_cpu.yaml
```

If Python 3.12 or `venv` is missing:

```bash
sudo apt update
sudo apt install python3.12 python3.12-venv
```

Uses the included `mobility_v1` dataset. Training automatically evaluates the model on the test set. Outputs are saved under `tf/src/experiments_outputs/`. For separate evaluation, use the same config changes and command shown above.

## RealWorld accelerometer + gyroscope

The `training_config_realworld_imu.yaml` example trains on six synchronized channels (accelerometer X/Y/Z followed by gyroscope X/Y/Z). It preserves the existing 24-sample window. The loader accepts a merged CSV with headers `acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, activity` and a `subject` column. `recording` is optional. Common axis aliases are accepted; adapt the headers in `realworld_imu.py` if the downloaded repository uses other names. Each row must contain both sensors from the same sensor location and timestamp.

Set `dataset.class_names` to the activity strings exactly as present in the CSV, and set `training_path` to that CSV. Use the chosen window length in `model.input_shape` (the example uses 24 samples). Windows are built within activity runs and sensor recordings, then split by subject to keep a subject out of other partitions. The loader requires a subject column, or a recording column when there is no subject column.

The existing gravity rotation/suppression option is disabled for this data because it is accelerometer-specific. Fine-tuning can be run with `python -m human_activity_recognition.tf.src.training.train_realworld_imu --csv PATH --source-model PATH --classes walking,running,climbing_stairs,standing --source-classes Stationary,Walking,Jogging,Biking --output-dir OUTPUT`. It writes best/last checkpoints, training history, normalization values, per-class metrics, and confusion matrices. Use the same saved training-set mean and standard deviation during deployed inference.

The repository's ST-IGN is a channels-last Conv2D model with shape `(batch, time, axes, 1)`, not a channels-first Conv1D. Its temporal convolution shares weights across axis positions. The transfer helper copies those convolution weights, maps flattened accelerometer features into the six-axis model, and zeroes the added gyroscope feature connections. The output classifier is transferred only when the source and target activity sets match; otherwise it remains newly initialized. The local pretrained checkpoint was trained on the mobility classes, so its classifier does not match the four example classes.

You can refer to following readme tutorials that provide typical examples of operation modes:

   - [training, chain_tb](./docs/README_TRAINING.md)
   - [evaluation](./docs/README_EVALUATION.md)
   - [benchmarking](./docs/README_BENCHMARKING.md)
   - [deployment](./docs/README_DEPLOYMENT.md)

All .yaml configuration examples can be found in the [config_file_examples](./config_file_examples/) folder.

The different values of the `operation_mode` attribute and the corresponding operations are described in the table below. In the names of the chain modes, 't' stands for training, and 'b' for benchmark on an STM32 board.

| operation_mode attribute | Operations |
|:---------------------------|:-----------|
| `training`| Train an HAR model |
| `evaluation` | Evaluate the accuracy of a pretrained float model on a test or validation dataset |
| `benchmarking` | Benchmark a pretrained float model on an STM32 board |
| `deployment`   | Deploy a pretrained float model on an STM32 board |
| `chain_tb`  | Sequentially: train, and then benchmark an HAR model |


## You don't know where to start? You feel lost?
Don't forget to follow our tuto below for a quick ramp up :
* [How to define and train my own model?](./docs/tuto/how_to_define_and_train_my_own_model.md)
* [How to fine tune a model on my own dataset?](./docs/tuto/how_to_finetune_a_model_zoo_model_on_my_own_dataset.md)
* [How can I quickly check the performance of my model using the dev cloud?](./docs/tuto/how_to_quickly_benchmark_the_performances_of_a_model.md)
* [How can I deploy my model?](./docs/tuto/how_to_deploy_a_model_on_a_target.md)

Remember that minimalistic yaml files are available [here](./config_file_examples/) to play with specific services, and that all pre-trained models in the [STM32 model zoo](https://github.com/STMicroelectronics/stm32ai-modelzoo/human_activity_recognition/) are provided with their configuration .yaml file used to generate them. These are very good starting points to start playing with!
