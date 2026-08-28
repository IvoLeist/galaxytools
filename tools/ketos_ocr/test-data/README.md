# Ketos OCR test data

The `ketos_train_recognition.arrow` fixture was compiled with Kraken 7.1 from
[`170025120000003,0074-lite.xml`](https://github.com/mittagessen/kraken/blob/main/tests/resources/170025120000003%2C0074-lite.xml)
and its referenced JPEG. This is the same compact PAGE fixture used by Kraken's
upstream recognition training smoke test.

`ketos_train_model.safetensors` contains the best weights produced by a
one-epoch CPU run over that Arrow dataset using the small upstream smoke-test
VGSL specification:

```text
[1,12,0,1 Cr3,3,8 S1(1x0)1,3]
```

This is a deliberately tiny [Kraken VGSL network](https://kraken.re/5.2/vgsl.html)
chosen to keep the smoke tests fast:

- `[1,12,0,1]` defines the input as `[batch, height, width, channels]`: one
  12-pixel-high, variable-width (`0`) grayscale channel.
- `Cr3,3,8` applies a 3 × 3 convolution with ReLU activation (`r`) and eight
  output channels.
- `S1(1x0)1,3` collapses the height into the channel dimension. It splits
  dimension 1 (height) into `1 × 0`, where `0` means infer the remaining
  factor, leaves the size-1 part in dimension 1, and moves the inferred part
  to dimension 3 (channels). After the convolution, this changes each
  width-wise feature vector from height 12 × 8 channels to height 1 × 96
  channels.

The specification intentionally omits an output block: during recognition
training Ketos appends the CTC output layer sized to the alphabet found in the
training data. This minimal convolution-and-reshape network is suitable for
testing the training workflow, not for producing an accurate OCR model.
