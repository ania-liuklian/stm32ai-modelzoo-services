"""Weight transfer for extending the repository's three-axis ST-IGN model."""
import numpy as np


def transfer_st_ign_weights(source, target, source_classes, target_classes):
    """Copy compatible ST-IGN weights and zero the added gyroscope feature path.

    The repository model is Conv2D/channels-last with input (time, axes, 1).
    Its first convolution shares one temporal kernel across axes; axes are not
    Conv2D input channels. After widening axes from 3 to 6, the flattened
    Dense input grows. This copies each old axis's Dense weights to its matching
    accelerometer axis and leaves the three gyro-axis connections at zero.
    """
    source_shape = tuple(int(x) for x in source.input_shape[1:])
    target_shape = tuple(int(x) for x in target.input_shape[1:])
    if len(source_shape) != 3 or len(target_shape) != 3:
        raise ValueError("Expected channels-last ST-IGN tensors (time, axes, 1).")
    if source_shape[0] != target_shape[0] or source_shape[2] != 1 or target_shape[2] != 1:
        raise ValueError(f"Only axis-count expansion is supported: {source_shape} -> {target_shape}.")
    if source_shape[1] != 3 or target_shape[1] != 6:
        raise ValueError(f"Expected three accelerometer axes expanded to six axes: {source_shape} -> {target_shape}.")

    transferred, skipped = [], []
    src_layers = [layer for layer in source.layers if layer.get_weights()]
    dst_layers = [layer for layer in target.layers if layer.get_weights()]
    if len(src_layers) != len(dst_layers):
        raise ValueError("Source and target layer topology differs; expected the same ST-IGN layers.")

    same_labels = set(source_classes) == set(target_classes) and len(source_classes) == len(target_classes)
    for index, (src, dst) in enumerate(zip(src_layers, dst_layers)):
        old = src.get_weights()
        new = dst.get_weights()
        is_classifier = index == len(src_layers) - 1
        if is_classifier and not same_labels:
            skipped.append(dst.name)
            continue

        if len(old) != len(new):
            skipped.append(dst.name)
            continue
        mapped = []
        valid = True
        for old_weight, new_weight in zip(old, new):
            if old_weight.shape == new_weight.shape:
                mapped.append(old_weight.copy())
                continue
            # The Dense layer after Flatten expands by 2x as the axis width
            # expands from three to six. Keep old accel-axis weights in place;
            # zero the three adjacent gyro-axis blocks.
            if (old_weight.ndim == 2 and new_weight.ndim == 2 and
                    old_weight.shape[1] == new_weight.shape[1] and
                    new_weight.shape[0] == old_weight.shape[0] * 2):
                old_axes, new_axes = source_shape[1], target_shape[1]
                conv_filters = source.layers[1].filters
                old_steps = old_weight.shape[0] // (old_axes * conv_filters)
                new_steps = new_weight.shape[0] // (new_axes * conv_filters)
                if (old_steps != new_steps or
                        old_weight.shape[0] != old_steps * old_axes * conv_filters or
                        new_weight.shape[0] != new_steps * new_axes * conv_filters):
                    valid = False
                    break
                old_view = old_weight.reshape(old_steps, old_axes, conv_filters, old_weight.shape[1])
                new_view = np.zeros((new_steps, new_axes, conv_filters, new_weight.shape[1]),
                                    dtype=new_weight.dtype)
                new_view[:, :old_axes] = old_view
                mapped.append(new_view.reshape(new_weight.shape))
            else:
                valid = False
                break

        if not valid:
            skipped.append(dst.name)
            continue
        # Reorder the classifier output columns when labels are the same set.
        if is_classifier and same_labels:
            permutation = [source_classes.index(name) for name in target_classes]
            mapped[0] = mapped[0][:, permutation]
            if len(mapped) > 1:
                mapped[1] = mapped[1][permutation]
        dst.set_weights(mapped)
        transferred.append(dst.name)
    return {"transferred": transferred, "skipped": skipped,
            "classifier_transferred": same_labels}
