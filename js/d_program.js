const tf = require('tfjs-node');
const { layers } = require('tfjs-node');

function convertTextToVector(text) {
    text = text.replace(' ', '').toLowerCase();
    const exp_d = new Array(64).fill(0);
    for (let i = 0; i < text.length; i++) {
        exp_d[i] = 1 / text.charCodeAt(i);
    }
    return exp_d;
}

function siameseNetwork(inputDim) {
    const input_a = tf.keras.Input({ shape: [inputDim] });
    const input_b = tf.keras.Input({ shape: [inputDim] });

    const sharedNetwork = tf.keras.Sequential([
        layers.Dense(128, { activation: 'relu' }),
        layers.Dense(128, { activation: 'relu' })
    ]);

    const encoded_a = sharedNetwork(input_a);
    const encoded_b = sharedNetwork(input_b);

    const distance = tf.keras.layers.Lambda(
        x => tf.math.abs(x[0].sub(x[1]))
    )([encoded_a, encoded_b]);

    const output = layers.Dense(1, { activation: 'sigmoid' })(distance);

    const model = tf.keras.Model({ inputs: [input_a, input_b], outputs: output });

    return model;
}

const model = siameseNetwork(64);
model.compile({ optimizer: 'adam', loss: 'binary_crossentropy' });

function forecast(n1, n2) {
    const input_a = tf.tensor([convertTextToVector(n1)]);
    const input_b = tf.tensor([convertTextToVector(n2)]);
    const result = model.predict([input_a, input_b]);
    return result.arraySync()[0];
}

async function trainModel() {
    const convert_funtions = require('./convert_functions');

    const input_a = convert_funtions.input_a;
    const input_b = convert_funtions.input_b;
    const labels = convert_funtions.labels;

    await model.fit([input_a, input_b], labels, { epochs: 500, batchSize: 32 });
    await model.save('models/model7');
}

async function loadModel() {
    await model.loadWeights('models/model7/model.json');
}

async function main() {
    const examples = [
        ['Формис II (жен)', 'Ledec nad Sazavou'],
        ['Формис II (жен)', 'Formis-2 (w)'],
        ['Real Sociedad (Nicolas_Rage)', 'Реал Сосьедад (Nicolas_Rage)'],
        ['Parentini Vallega Montebruno G/Ruggeri J', 'Parentini Vallega Montebruno / Ruggeri'],
        ['Formis-2 (w)', 'Формис II (жен)'],
        ['Formis-2 (w)', 'Formis-2 (w)'],
        ['спрпропро', 'нггвгнынгоен'],
        ['Real Sociedad (Nicolas_Rage)', 'Real Sociedad (Nicolas_Rage) Esports'],
        ['Austin Peay (w)', 'Austin Peay Women'],
        ['Ferroviaria San Paolo', 'Ферровиария СП'],
        ['Ferroviaria San Paolo', 'Ferroviaria SP'],
        ['Elizabeth Ionescu', 'Elizabeth Ionescu (USA)']
    ];

    for (const [n1, n2] of examples) {
        console.log(n1, '+', n2, ' -> ', forecast(n1, n2));
    }

    let n1, n2;
    [n1, n2] = input('Введите имена: ').split('!');
    while (n1 !== '0' && n2 !== '0') {
        console.log(forecast(n1, n2));
        [n1, n2] = input('Введите имена: ').split('!');
    }
}

async function run() {
    const a = 1;
    if (a) {
        await trainModel();
        await model.save('models/model7');
    } else {
        await loadModel();
    }
    await main();
}

run();
