const brain = require('brain.js');
const fs = require('fs');
const sqlite3 = require('sqlite3').verbose();
const axios = require('axios');

const db = new sqlite3.Database('../data/names.db');

// class AppDAO {
//     constructor(dbFilePath) {
//         this.db = new sqlite3.Database(dbFilePath, (err) => {
//             if (err) {
//                 console.log('Could not connect to database', err)
//             } else {
//                 console.log('Connected to database')
//             }
//         })
//     }
//
//     run(sql, params = []) {
//         return new Promise((resolve, reject) => {
//             this.db.run(sql, params, function (err) {
//                 if (err) {
//                     console.log('Error running sql ' + sql)
//                     console.log(err)
//                     reject(err)
//                 } else {
//                     resolve({id: this.lastID})
//                 }
//             })
//         })
//     }
// }

var dataset_for_train_model = [];
var page = 2382;

// const dao = new AppDAO('./data/names.db');

async function convert_game_name(name1, name2, b1, b2, st) {
    db.run(
        'INSERT INTO data(name1, name2, status, b1, b2) VALUES(?, ?, ?, ?, ?)',
        [name1, name2, st, b1, b2], (e, r)=>{})
    db.run('commit');
    var exp_d = new Array(64).fill(0);
    exp_d[0] = b1;
    exp_d[63] = b2;
    for (let i = 0; i < name1.length; i++) {
        exp_d[i + 1] = name1[i].charCodeAt(0);
    }
    for (let i = 0; i < name2.length; i++) {
        exp_d[62 - i] = name2[i].charCodeAt(0);
    }
    dataset_for_train_model.push({
        input: exp_d,
        output: [st]
    })
    // return exp_d;
}


async function add_data(response) {
    var dataset = response.data.data;
    // console.log(dataset);
    for (let obj_num = 0; obj_num < dataset.length; obj_num++) {
        let obj = dataset[obj_num];
        var team_count = obj.teams.length;
        let teams = obj.teams;
        if (obj.checkedAt) {
            for (let team1_num = 0; team1_num < team_count; team1_num++) {
                for (let team2_num = team1_num; team2_num < team_count; team2_num++) {
                    convert_game_name(teams[team1_num].name, teams[team2_num].name, teams[team1_num].bookieId, teams[team2_num].bookieId, 1);
                }
            }
        }

        for (let obj1_num = obj_num + 1; obj1_num < dataset.length; obj1_num++) {
            let obj1 = dataset[obj1_num];

            let teams1 = obj1.teams;
            let team_count1 = obj1.teams.length;
            for (let team1_num1 = 0; team1_num1 < team_count; team1_num1++) {
                for (let team2_num1 = team1_num1; team2_num1 < team_count1; team2_num1++) {
                    convert_game_name(teams[team1_num1].name, teams1[team2_num1].name, teams[team1_num1].bookieId, teams1[team2_num1].bookieId, 0)
                }
            }

        }
    }
    if (page > 2400) {
        console.log(page, '/', 2400)
        page -= 1;
        add_data(await axios.get(`https://sm.livesport.tools/api/game-manager/team_groups?page=${page}`));
    } else {
        console.log(dataset_for_train_model.length)
        // create_model(dataset_for_train_model);
    }

    // console.log(dataset_for_train_model);Формис
}


axios.get(`https://sm.livesport.tools/api/game-manager/team_groups?page=${page}`).then(add_data);

function create_model(data) {
    var net = new brain.NeuralNetwork();


    net.train(data, {
        iterations: 1000,   // maximum training iterations
        log: true,           // console.log() progress periodically
        logPeriod: 1,       // number of iterations between logging
    });


    console.log(net.run(convert_game_name('Formis', 'Формис', 1, 2)))
    let wstream = fs.createWriteStream('../models/model_name1.json');
    wstream.write(JSON.stringify(net.toJSON(), null, 2));
    wstream.end();

    console.log('MNIST dataset with Brain.js train done.')
}


// create_model();