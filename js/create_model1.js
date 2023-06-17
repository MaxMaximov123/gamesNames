const brain = require("brain.js");
const fs = require("fs");
const sqlite3 = require('sqlite3').verbose();

const db = new sqlite3.Database('./data/names.db');

var dataset_for_train_model = [];

function convert_game_name(name1, name2, st, b1, b2) {
    var exp_d = new Array(64).fill(0);
    exp_d[0] = b1;
    exp_d[63] = b2;
    // for (let i = 0; i < name1.length; i++) {
    //     exp_d[i + 1] = name1[i].charCodeAt(0);
    // }
    // for (let i = 0; i < name2.length; i++) {
    //     exp_d[62 - i] = name2[i].charCodeAt(0);
    // }
    dataset_for_train_model.push({
        input: [name1, name2],
        output: [st]
    })
    // console.log(dataset_for_train_model[dataset_for_train_model.length - 1])
    return exp_d;
}

function create_model(data) {
    var net = new brain.recurrent.LSTM();

    console.log(new Date().toLocaleTimeString(
        'en-GB', {
            hour: "numeric",
            minute: "numeric",
            second: "numeric"}), 'START TRAINING')
    net.train(data.slice(0, 10000), {
        iterations: 1000,   // maximum training iterations
        log: (it)=>{console.log(new Date().toLocaleTimeString(
            'en-GB', {
                hour: "numeric",
                minute: "numeric",
                second: "numeric"}), it)},           // console.log() progress periodically
        logPeriod: 1,       // number of iterations between logging
    })
    console.log(net.run(['Спартак', 'Спартак (ж)']));
    fs.writeFileSync('./models/model_name4.json', JSON.stringify(net.toJSON()));

    console.log('MNIST dataset with Brain.js train done.')



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