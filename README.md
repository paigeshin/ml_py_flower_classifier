# ml_py_flower_classifier

# 주의점

python 2.7에서 실행할 것

# Code

```python

import coremltools

#p1 - caffemodel
#p2 - file that describes the architecture of the caffe model
caffe_model = ('oxford102.caffemodel', 'deploy.prototxt')

labels = 'flower-labels.txt'

coreml_model = coremltools.converters.caffe.convert(
    caffe_model,
    class_labels = labels,
    image_input_names = 'data'
)

coreml_model.save('FlowerClassifier.mlmodel')


```
