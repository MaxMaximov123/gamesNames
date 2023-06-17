const brain = require("brain.js");
const fs = require("fs");
const sqlite3 = require('sqlite3').verbose();

const db = new sqlite3.Database('./data/names.db');

var dataset_for_train_model = [];

function convert_game_name(name1, name2, st, b1, b2) {
    var exp_d = new Array(64).fill(0);
    exp_d[0] = b1;
    exp_d[63] = b2;
    for (let i = 0; i < name1.length; i++) {
        exp_d[i + 1] = 1 / name1[i].charCodeAt(0);
    }
    for (let i = 0; i < name2.length; i++) {
        exp_d[62 - i] = 1 / name2[i].charCodeAt(0);
    }
    dataset_for_train_model.push({
        input: exp_d,
        output: [st]
    })
    return exp_d;
}

function create_model(data) {
    var net = new brain.NeuralNetwork();

    console.log('START TRAINING')
    net.trainAsync(data, {
        iterations: 100,   // maximum training iterations
        log: (it)=>{console.log(new Date().toLocaleTimeString(
            'en-GB', {
                hour: "numeric",
                minute: "numeric",
                second: "numeric"}), it)},           // console.log() progress periodically
        logPeriod: 1,       // number of iterations between logging
    }).then((res) => {
        console.log(net.run(convert_game_name('Спартак', 'Спартак (ж)', 0, 1, 2)))
        let wstream = fs.createWriteStream('./models/model_name1.json');
        wstream.write(JSON.stringify(net.toJSON(), null, 2));
        wstream.end();

        console.log('MNIST dataset with Brain.js train done.')
    });



}


db.all(`SELECT * FROM data`, [], function (err, rows) {
    if (err) {
        console.warn(err);
        throw err;
    }
    rows.forEach(function (row) {
        convert_game_name(
            row.name1,
            row.name2,
            row.status,
            row.b1,
            row.b2
        );
    });
    create_model(dataset_for_train_model);
});

db.close();