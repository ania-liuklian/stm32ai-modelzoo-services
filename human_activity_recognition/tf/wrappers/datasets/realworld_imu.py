from common.registries.dataset_registry import DATASET_WRAPPER_REGISTRY
from human_activity_recognition.tf.src.datasets.realworld_imu import load_realworld_imu
from human_activity_recognition.tf.src.datasets.utils import prepare_kwargs_for_dataloader


@DATASET_WRAPPER_REGISTRY.register(framework='tf', dataset_name='realworld_imu',
                                   use_case="human_activity_recognition")
def get_realworld_imu(cfg):
    args = prepare_kwargs_for_dataloader(cfg)
    train_ds, val_ds, test_ds = load_realworld_imu(
        training_path=args['training_path'],
        class_names=args['class_names'],
        input_shape=args['input_shape'],
        validation_split=args['validation_split'],
        test_split=args['test_split'],
        batch_size=args['batch_size'],
        seed=args['seed'],
        normalization=args['normalization'],
        to_cache=args['to_cache'])
    return {'train': train_ds, 'valid': val_ds, 'test': test_ds}
