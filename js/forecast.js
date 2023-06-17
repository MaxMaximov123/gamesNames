const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');


function convertTextToVector(text) {
    const exp_d = new Array(64).fill(0);
    for (let i = 0; i < text.length; i++) {
        exp_d[i] = 1 / text.charCodeAt(i);
    }
    return exp_d;
}

function siameseNetwork(inputDim) {
    const input_a = tf.input({ shape: [inputDim] });
    const input_b = tf.input({ shape: [inputDim] });

    const sharedNetwork = tf.sequential();
    sharedNetwork.add(tf.layers.dense({ units: 128, activation: 'relu' }));
    sharedNetwork.add(tf.layers.dense({ units: 128, activation: 'relu' }));

    const encoded_a = sharedNetwork.apply(input_a);
    const encoded_b = sharedNetwork.apply(input_b);

    const distance = tf.layers.lambda(
        (inputs) => tf.math.abs(inputs[0].sub(inputs[1]))
    ).apply([encoded_a, encoded_b]);

    const output = tf.layers.dense({ units: 1, activation: 'sigmoid' }).apply(distance);

    const model = tf.model({ inputs: [input_a, input_b], outputs: output });
    return model;
}

const model = siameseNetwork(64);

model.compile({ optimizer: 'adam', loss: 'binaryCrossentropy' });

(async () => {
    await model.loadWeights('file://../models/model3/model.json');

    function forecast(n1, n2) {
        const input_a = tf.tensor2d([convertTextToVector(n1)]);
        const input_b = tf.tensor2d([convertTextToVector(n2)]);
        const result = model.predict([input_a, input_b]);
        return result.arraySync()[0][0];
    }

    console.log(forecast('Formis', 'Formis'));
})();
