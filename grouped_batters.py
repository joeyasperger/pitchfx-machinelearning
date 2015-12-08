import numpy as np
import json
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn import svm
from pymongo import MongoClient
from bson.objectid import ObjectId
from collections import Counter
from sklearn.feature_extraction import DictVectorizer


events =  {
    'Strikeout': 1,
    'Strikeout - DP': 1, 

    'Pop Out': .8, 
    'Flyout': .7, 

    'Groundout': .7, 
    'Forceout': .7, 
    'Grounded Into DP': .7, 
    'Double Play': .7, 
    'Triple Play': .7,
    'Fielders Choice': .7,
    'Fielders Choice Out': .7, 
    'Field Error': .7, 

    'Lineout': .2, 
    'Runner Out': .1, 


    'Sac Fly': -.3,
    'Sac Fly DP': -.3,  

    'Hit By Pitch': -1, 
    'Walk': -1, 

    'Single': -1, 
    'Double': -1.3, 
    'Triple': -1.5, 
    'Home Run': -2, 
 
    'Sac Bunt': 0, 
    'Batter Interference': 0, 
    'Intent Walk': 0, 
    'Bunt Pop Out': 0, 
    'Bunt Groundout': 0,
    'Catcher Interference': 0, 
    'Bunt Lineout': 0, 
    'Fan interference': 0, 
    'Sacrifice Bunt DP': 0
}

def getFeatures(pitch, cluster_num):
    features = {}
    feature_vec = []
    features["start_speed"] = pitch["start_speed"]
    features["start_speed^2"] = pitch["start_speed"] ** 2
    features["px^2"] = pitch["px"] ** 2
    if pitch["px"] == 'R':
        features["px"] = pitch["px"]
    else:
        features["px"] = -pitch["px"]
    features["pfx_x^2"] = pitch["pfx_x"] ** 2
    features["pfx_x"] = pitch["pfx_x"]
    features["pfx_z^2"] = pitch["pfx_z"] ** 2
    features["pfx_z"] = pitch["pfx_z"]
    features["pz-mid"] = pitch["pz"] - (pitch["sz_top"] + pitch["sz_bot"])/2
    features["pz-mid^2"] = (pitch["pz"] - (pitch["sz_top"] + pitch["sz_bot"])/2) ** 2
    features["break_angle"] = pitch["break_angle"]
    features["break_length"] = pitch["break_length"]
    features["break_y"] = pitch["break_y"]
    features["px*pz-mid"] = features["px"] * features["pz-mid"]
    features["px^2*(pz-mid)^2"] = (features["px"] ** 2) * (features["pz-mid"] ** 2)

    if pitch["pitch_type"] == "FF":
        features["FF"] = 1
    else:
        features["FF"] = 0
    if pitch["pitch_type"] == "CU":
        features["CU"] = 1
    else:
        features["CU"] = 0
    if pitch["pitch_type"] == "CH":
        features["CH"] = 1
    else:
        features["CH"] = 0
    if pitch["prev_type"] == "FF":
        features["prev_FF"] = 1
    else:
        features["prev_FF"] = 0
    if pitch["prev_type"] == "CU":
        features["prev_CU"] = 1
    else:
        features["prev_CU"] = 0
    if pitch["prev_type"] == "CH":
        features["prev_CH"] = 1
    else:
        features["prev_CH"] = 0

    attribs = ["start_speed", "start_speed^2", "px^2", "px", "pfx_x^2", "pfx_x", 
        "pfx_z^2", "pfx_z", "pz-mid", "pz-mid^2", "break_angle", "break_length", "break_y",
        "px*pz-mid", "px^2*(pz-mid)^2"]
    for attrib in attribs:
        for pitch_type in ["prev_FF", "prev_CU", "prev_CH", "FF", "CU", "CH"]:
            features[pitch_type + "_" + attrib] = features[pitch_type] * features[attrib]

    zones = ['TL', 'TM', 'TR', 'ML', 'MM', 'MR', 'BL', 'BM', 'BR']
    for zone in zones:
        features['zone_' + zone] = 0
        features['prev_zone_' + zone] = 0
    current_zone = getZone(pitch['px'], pitch['pz'], pitch['sz_top'], pitch['sz_bot'])
    features['zone_' + current_zone] = 1
    if pitch['prev_px'] != None:
        prev_zone = getZone(pitch['prev_px'], pitch['prev_pz'], pitch['sz_top'], pitch['sz_bot'])
        features['prev_zone_' + prev_zone] = 1
    for zone1 in zones:
        for zone2 in zones:
            features[zone1 + '*prev=' + zone2] = features['zone_' + zone1] * features['prev_zone_' + zone2]
    feature_keys = features.keys()
    for key in feature_keys:
        features['cluster%d_' % cluster_num + key] = features[key]
    return features

def getZone(px, pz, sz_top, sz_bot):
    if pz - sz_bot < (sz_top - sz_bot)/3.0:
        # lower third
        if px < -0.236:
            return 'TL'
        if px < 0.236:
            return 'TM'
        else:
            return 'TR'
    elif pz - sz_bot < (sz_top - sz_bot)/3.0 * 2:
        #middle third
        if px < -0.236:
            return 'ML'
        if px < 0.236:
            return 'MM'
        else:
            return 'MR'
    else:
        #top third
        if px < -0.236:
            return 'BL'
        if px < 0.2363:
            return 'BM'
        else:
            return 'BR'


def train(reg, scaler, num_pitches, vec, players):
    client = MongoClient('localhost', 27017)
    db = client["pitchfx"]
    x = []
    y = []
    num_added = 0
    # for pitch in db.pitches.find({"pitch_type":"FF", "batter":{"$in": batters}}, limit=100000):
    for pitch in db.pitches.find({"pitch_type":{"$in": ["FF", "CH", "CU"]}, "type":{"$in": ["S", "X"]}}, limit=num_pitches):
        cluster = players.get((pitch['batter'], pitch['year']))
        if cluster == None:
            continue
        feature_vec = getFeatures(pitch, cluster)
        if pitch["type"] == 'S':
            result = 1
        if pitch["type"] == 'B':
            continue
        if pitch["type"] == 'X':
            result = events[pitch["event"]]
        x.append(feature_vec)
        y.append(result)
        num_added += 1
    print num_added, ' pitches added'
    scaler.fit(vec.fit_transform(x).toarray())
    reg.fit(scaler.transform(vec.transform(x).toarray()),y)
    print 'finished fitting data'

def testTrain(reg, scaler, num_pitches, vec, players):
    client = MongoClient('localhost', 27017)
    db = client["pitchfx"]
    num_added = 0
    total_error = 0
    total_error_sq = 0
    total_maj_err = 0
    total_maj_err_sq = 0
    # for pitch in db.pitches.find({"pitch_type":"FF", "batter":{"$in": batters}}, limit=100000):
    for pitch in db.pitches.find({"pitch_type":{"$in": ["FF", "CH", "CU"]}, "type":{"$in": ["S", "X"]}}, limit=num_pitches):
        cluster = players.get((pitch['batter'], pitch['year']))
        if cluster == None:
            continue
        feature_vec = getFeatures(pitch, cluster)
        if pitch["type"] == 'S':
            result = 1
        if pitch["type"] == 'B':
            continue
        if pitch["type"] == 'X':
            result = events[pitch["event"]]
        #print reg.predict(scaler.transform(vec.transform(feature_vec).toarray()))[0]
        error = reg.predict(scaler.transform(vec.transform(feature_vec).toarray()))[0] - result
        #print 'prediction: ', reg.predict(scaler.transform([feature_vec]))[0], ' result: ', result 
        num_added += 1    
        total_error += abs(error)
        total_error_sq += error ** 2
        majority_error = 1-result
        total_maj_err += abs(majority_error)
        total_maj_err_sq += majority_error ** 2
    print num_added, ' pitches tested'
    print 'Train Error = ', total_error, ' error per pitch = ', total_error/float(num_added), 'squared error =', total_error_sq/float(num_added)
    print 'Train Error (majority algorithm) = ', total_maj_err, ' error per pitch = ', total_maj_err/float(num_added), 'squared error =', total_maj_err_sq/float(num_added)

def test(reg, scaler, num_pitches, vec, players):
    client = MongoClient('localhost', 27017)
    db = client["pitchfx"]
    num_added = 0
    total_error = 0
    total_error_sq = 0
    total_maj_err = 0
    total_maj_err_sq = 0
    # for pitch in db.pitches.find({"pitch_type":"FF", "batter":{"$in": batters}}, limit=100000, skip=100000):
    for pitch in db.pitches.find({"pitch_type":{"$in": ["FF", "CH", "CU"]}, "type":{"$in": ["S", "X"]}}, limit=100000, skip=num_pitches):
        cluster = players.get((pitch['batter'], pitch['year']))
        if cluster == None:
            continue
        feature_vec = getFeatures(pitch, cluster)
        if pitch["type"] == 'S':
            result = 1
        if pitch["type"] == 'B':
            continue
        if pitch["type"] == 'X':
            result = events[pitch["event"]]
        error = reg.predict(scaler.transform(vec.transform(feature_vec).toarray()))[0] - result
        #print 'prediction: ', reg.predict(scaler.transform([feature_vec]))[0], ' result: ', result 
        num_added += 1
        total_error += abs(error)
        total_error_sq += error ** 2
        majority_error = 1-result
        total_maj_err += abs(majority_error)
        total_maj_err_sq += majority_error ** 2
    print num_added, ' pitches tested'
    print 'Test Error = ', total_error, ' error per pitch = ', total_error/float(num_added), 'squared error =', total_error_sq/float(num_added)
    print 'Test Error (majority algorithm) = ', total_maj_err, ' error per pitch = ', total_maj_err/float(num_added), 'squared error =', total_maj_err_sq/float(num_added)

def kmeans_features(player, year):
    features = {}
    features['avg'] = player['avg_%04d' % year]
    features['hr'] = player['hr_%04d' % year]
    features['slg'] = player['slg_%04d' % year]
    features['so%'] = float(player['so_%04d' % year]) / (player['ab_%04d' % year] + player['bb_%04d' % year])
    features['bb%'] = float(player['bb_%04d' % year]) / (player['ab_%04d' % year] + player['bb_%04d' % year])
    return features

def classifyWithKmeans(num_clusters):
    client = MongoClient('localhost', 27017)
    db = client["pitchfx"]
    x = []

    for player in db.players.find():
        for year in range(2008, 2016):
            if player.get('h_%d' % year) == None or player.get('ab_%d' % year) < 100:
                continue
            x.append(kmeans_features(player, year))
    kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10, random_state=1000)

    vec = DictVectorizer()
    scaler = StandardScaler()
    scaler.fit(vec.fit_transform(x).toarray())
    kmeans.fit(scaler.transform(vec.transform(x).toarray()))
    print json.dumps(vec.inverse_transform(scaler.inverse_transform(kmeans.cluster_centers_)), indent=4)
    for i in range(0,8):
        print 'cluster %d:' % i, list(kmeans.labels_).count(i)
    return (kmeans, scaler, vec)

# classifications = classifyPlayers()
# scaler = StandardScaler()
# reg = SGDRegressor(loss='squared_loss', n_iter=5)
# regressors = {
#     'highVol': SGDRegressor(loss='squared_loss', n_iter=5),
#     'balanced': SGDRegressor(loss='squared_loss', n_iter=5),
#     'defensive': SGDRegressor(loss='squared_loss', n_iter=5),
#     'inPlay': SGDRegressor(loss='squared_loss', n_iter=5)
# }
# train(regressors, scaler, classifications, reg)
# test(regressors, scaler, classifications, reg)

# vec = DictVectorizer()
# scaler = StandardScaler()
# num_iters = 100
# reg = SGDRegressor(loss='squared_loss', n_iter=num_iters, verbose=2, penalty='l2', alpha= 0.001, learning_rate="invscaling", eta0=0.002, power_t=0.4)
# num_pitches = 100000
# print num_pitches, 'pitches'
# print 'training with num iters = ', num_iters
# train(reg, scaler, None, num_pitches, vec)
# print reg.coef_
# print json.dumps(vec.inverse_transform([reg.coef_]), sort_keys=True, indent=4)
# testTrain(reg, scaler, None, num_pitches, vec)
# test(reg, scaler, None, num_pitches, vec)


client = MongoClient('localhost', 27017)
db = client["pitchfx"]
(kmeans, kmeans_scaler, kmeans_vec) = classifyWithKmeans(8)
players = {}
for player in db.players.find():
    for year in range(2008, 2016):
        if player.get('h_%d' % year) == None or player.get('ab_%d' % year) < 100:
            continue
        players[(player["player_id"], year)] = kmeans.predict(kmeans_scaler.transform(kmeans_vec.transform(kmeans_features(player, year)).toarray()))[0]
print 'finished mapping players'
vec = DictVectorizer()
scaler = StandardScaler()
num_iters = 100
reg = SGDRegressor(loss='squared_loss', n_iter=num_iters, verbose=2, penalty='l2', alpha= 0.001, learning_rate="invscaling", eta0=0.002, power_t=0.4)
num_pitches = 300000
print num_pitches, 'pitches'
print 'training with num iters = ', num_iters
train(reg, scaler, num_pitches, vec, players)
print reg.coef_
print json.dumps(vec.inverse_transform([reg.coef_]), sort_keys=True, indent=4)
testTrain(reg, scaler, num_pitches, vec, players)
test(reg, scaler, num_pitches, vec, players)


# client = MongoClient('localhost', 27017)
# db = client["pitchfx"]
# num_clusters = 8
# (kmeans, scaler) = classifyWithKmeans(8)
# data = []
# regs = []
# scalers = []
# for i in range(num_clusters):
#     data.append([])
#     regs.append(SGDRegressor(loss='squared_loss', n_iter=5))
#     scalers.append(StandardScaler())
# for player in db.players.find():
#     if player.get('h_2014') == None or player.get('h_2014') < 100:
#         continue
#     cluster = kmeans.predict(scaler.transform([kmeans_features(player)]))[0]
#     data[cluster].append(player["player_id"])
# print "1111111"
# for i in range(num_clusters):
#     train(regs[i], scalers[i], data[i])
#     testTrain(regs[i], scalers[i], data[i])
#     test(regs[i], scalers[i], data[i])


