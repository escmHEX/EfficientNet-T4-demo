import os
from flask import Flask, request, render_template, redirect
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0, ResNet50, DenseNet169, InceptionV3, Xception
from tensorflow.keras.applications.efficientnet import decode_predictions as decode_efficientnet, preprocess_input as preprocess_efficientnet
from tensorflow.keras.applications.resnet50 import decode_predictions as decode_resnet, preprocess_input as preprocess_resnet
from tensorflow.keras.applications.densenet import decode_predictions as decode_densenet, preprocess_input as preprocess_densenet
from tensorflow.keras.applications.inception_v3 import decode_predictions as decode_inception, preprocess_input as preprocess_inception
from tensorflow.keras.applications.xception import decode_predictions as decode_xception, preprocess_input as preprocess_xception
from tensorflow.keras.preprocessing import image
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'

models = {
    'EfficientNetB0': (EfficientNetB0(weights='imagenet'), preprocess_efficientnet, decode_efficientnet),
    'ResNet50': (ResNet50(weights='imagenet'), preprocess_resnet, decode_resnet),
    'DenseNet169': (DenseNet169(weights='imagenet'), preprocess_densenet, decode_densenet),
    'InceptionV3': (InceptionV3(weights='imagenet'), preprocess_inception, decode_inception),
    'Xception': (Xception(weights='imagenet'), preprocess_xception, decode_xception)
}

def prepare_image(img_path, target_size):
    img = image.load_img(img_path, target_size=target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return x

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.makedirs(app.config['UPLOAD_FOLDER'])
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            model_choices = request.form.getlist("model_choices[]")
            model_choices = ['EfficientNetB0'] + model_choices

            results = {}
            for model_name in model_choices:
                model, preprocess, decode = models[model_name]
                target_size = (299, 299) if model_name in ['InceptionV3', 'Xception'] else (224, 224)
                x = prepare_image(filepath, target_size)
                x_preprocessed = preprocess(x.copy())

                flops = get_flops(model, list(x_preprocessed.shape[1:]))
                preds = model.predict(x_preprocessed)
                decoded_preds = decode(preds, top=1)[0][0]

                results[model_name] = {
                    'class': decoded_preds[1],
                    'flops': round(flops / 1e9, 1),  # FLOPS a GFLOPS y redondeado a 1 decimal
                    'num_params': round(model.count_params()/1e6, 1)
                }
            
            return render_template("results.html", results=results, img_path=filepath)

    return render_template("index.html")

def get_flops(model, input_shape):
    concrete = tf.function(lambda inputs: model(inputs))
    concrete_func = concrete.get_concrete_function(
        tf.TensorSpec([1] + input_shape, model.inputs[0].dtype))

    frozen_func = convert_variables_to_constants_v2(concrete_func)
    graph_def = frozen_func.graph.as_graph_def()

    with tf.Graph().as_default() as graph:
        tf.compat.v1.import_graph_def(graph_def, name="")
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(graph=graph,
                                              run_meta=run_meta, cmd='op', options=opts)
        return flops.total_float_ops

if __name__ == "__main__":
    app.run(debug=True)
