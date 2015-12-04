import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn import svm
from pymongo import MongoClient
from bson.objectid import ObjectId

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

def getFeaturesFF(pitch):
    feature_vec = []
    feature_vec.append(pitch["start_speed"])
    feature_vec.append(pitch["px"] ** 2)
    if pitch["batter_hand"] == 'R':
        feature_vec.append(pitch["px"])
    else:
        feature_vec.append(-pitch["px"])
    feature_vec.append((pitch["pz"] - (pitch["sz_top"] - pitch["sz_top"])) ** 2)
    feature_vec.append(pitch["pz"] - (pitch["sz_top"] - pitch["sz_top"]))
    feature_vec.append(pitch["pfx_x"] ** 2)
    feature_vec.append(pitch["pfx_x"])
    feature_vec.append(pitch["pfx_z"] ** 2)
    feature_vec.append(pitch["pfx_z"])
    return feature_vec

def getFeaturesCU(pitch):
    feature_vec = []
    feature_vec.append(pitch["start_speed"]) # positive
    feature_vec.append(pitch["px"] ** 2) # 
    if pitch["batter_hand"] == 'R':
        feature_vec.append(pitch["px"])
    else:
        feature_vec.append(-pitch["px"])
    feature_vec.append((pitch["pz"] - (pitch["sz_top"] - pitch["sz_top"])) ** 2) # matters a lot
    feature_vec.append(pitch["pz"] - (pitch["sz_top"] - pitch["sz_top"])) #lower is better, really negative weight
    feature_vec.append(pitch["pfx_x"] ** 2)
    feature_vec.append(pitch["pfx_x"])
    feature_vec.append(pitch["pfx_z"] ** 2)
    feature_vec.append(pitch["pfx_z"])
    feature_vec.append(pitch["break_angle"])
    feature_vec.append(pitch["break_length"])
    feature_vec.append(pitch["break_y"])
    return feature_vec

def getFeaturesCH(pitch):
    feature_vec = []
    feature_vec.append(pitch["start_speed"]) # positive
    feature_vec.append(pitch["px"] ** 2) # 
    if pitch["batter_hand"] == 'R':
        feature_vec.append(pitch["px"])
    else:
        feature_vec.append(-pitch["px"])
    feature_vec.append((pitch["pz"] - (pitch["sz_top"] - pitch["sz_top"])) ** 2) # matters a lot
    feature_vec.append(pitch["pz"] - (pitch["sz_top"] - pitch["sz_top"])) #lower is better, really negative weight
    feature_vec.append(pitch["pfx_x"] ** 2)
    feature_vec.append(pitch["pfx_x"])
    feature_vec.append(pitch["pfx_z"] ** 2)
    feature_vec.append(pitch["pfx_z"])
    feature_vec.append(pitch["break_angle"])
    feature_vec.append(pitch["break_length"])
    feature_vec.append(pitch["break_y"])
    return feature_vec


def train(regressors, scaler, classifications, reg):
    client = MongoClient('localhost', 27017)
    db = client["pitchfx"]
    xVals = {
        'highVol': [],
        'balanced': [],
        'defensive': [],
        'inPlay': []
    }
    yVals = {
        'highVol': [],
        'balanced': [],
        'defensive': [],
        'inPlay': []
    }
    x = []
    y = []
    for pitch in db.pitches.find({"pitch_type":"CH", "year":2014}, limit=50000):
        feature_vec = getFeaturesCH(pitch)
        if pitch["type"] == 'S':
            result = 1
        if pitch["type"] == 'B':
            continue
        if pitch["type"] == 'X':
            result = events[pitch["event"]]
        if pitch['batter'] in classifications['highVol']:
            xVals['highVol'].append(feature_vec)
            yVals['highVol'].append(result)
            x.append(feature_vec)
            y.append(result)
        elif pitch['batter'] in classifications['balanced']:
            xVals['balanced'].append(feature_vec)
            yVals['balanced'].append(result)
            x.append(feature_vec)
            y.append(result)
        elif pitch['batter'] in classifications['defensive']:
            xVals['defensive'].append(feature_vec)
            yVals['defensive'].append(result)
            x.append(feature_vec)
            y.append(result)
        elif pitch['batter'] in classifications['inPlay']:
            xVals['inPlay'].append(feature_vec)
            yVals['inPlay'].append(result)
            x.append(feature_vec)
            y.append(result)

    print len(y), ' pitches added'
    scaler.fit(x)
    reg.fit(scaler.transform(x),y)
    regressors['highVol'].fit(scaler.transform(xVals['highVol']),yVals['highVol'])
    regressors['balanced'].fit(scaler.transform(xVals['balanced']),yVals['balanced'])
    regressors['defensive'].fit(scaler.transform(xVals['defensive']),yVals['defensive'])
    regressors['inPlay'].fit(scaler.transform(xVals['inPlay']),yVals['inPlay'])


def test(regressors, scaler, classifications, reg):
    client = MongoClient('localhost', 27017)
    db = client["pitchfx"]
    num_added = 0
    total_error = 0
    total_errors = {
        'highVol': 0,
        'balanced': 0,
        'defensive': 0,
        'inPlay': 0
    }
    num_addeds = {
        'highVol': 0,
        'balanced': 0,
        'defensive': 0,
        'inPlay': 0
    }
    for pitch in db.pitches.find({"pitch_type":"CH", "year":2014}, limit=50000, skip=50000):
        feature_vec = getFeaturesCH(pitch)
        if pitch["type"] == 'S':
            result = 1
        if pitch["type"] == 'B':
            continue
        if pitch["type"] == 'X':
            result = events[pitch["event"]]
        if pitch['batter'] in classifications['highVol']:
            total_errors['highVol'] += abs(regressors['highVol'].predict(scaler.transform([feature_vec]))[0] - result)
            num_addeds['highVol'] += 1
        elif pitch['batter'] in classifications['balanced']:
            total_errors['balanced'] += abs(regressors['balanced'].predict(scaler.transform([feature_vec]))[0] - result)
            num_addeds['balanced'] += 1
        elif pitch['batter'] in classifications['defensive']:
            total_errors['defensive'] += abs(regressors['defensive'].predict(scaler.transform([feature_vec]))[0] - result)
            num_addeds['defensive'] += 1
        elif pitch['batter'] in classifications['inPlay']:
            total_errors['inPlay'] += abs(regressors['inPlay'].predict(scaler.transform([feature_vec]))[0] - result)
            num_addeds['inPlay'] += 1
        else:
            continue
            
        error = reg.predict(scaler.transform([feature_vec]))[0] - result
        #print 'prediction: ', reg.predict(scaler.transform([feature_vec]))[0], ' result: ', result 
        num_added += 1
        total_error += abs(error)
    print num_added, ' pitches tested'
    print 'Old method: Error = ', total_error, ' error per pitch = ', total_error/float(num_added)
    total_error_class = 0
    num_added_class = 0
    for key in ['highVol', 'balanced', 'defensive', 'inPlay']:
        print key +': Error = ', total_errors[key], ' error per pitch = ', total_errors[key]/float(num_addeds[key])
        total_error_class += total_errors[key]
        num_added_class += num_addeds[key]
    print 'Total with classifications: Error = ', total_error_class, ' error per pitch = ', total_error_class/float(num_added_class)

def kmeans_features(player):
    features = []
    features.append(player['avg_2014'])
    features.append(player['hr_2014'])
    features.append(player['slg_2014'])
    features.append(float(player['so_2014']) / (player['ab_2014'] + player['bb_2014']))
    features.append(float(player['bb_2014']) / (player['ab_2014'] + player['bb_2014']))
    return features

def classifyWithKmeans(num_clusters):
    client = MongoClient('localhost', 27017)
    db = client["pitchfx"]
    x = []
    for player in db.players.find():
        if player.get('h_2014') == None or player.get('h_2014') < 100:
            continue
        x.append(kmeans_features(player))
    kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10, random_state=1000)

    scaler = StandardScaler()
    scaler.fit(x)
    kmeans.fit(scaler.transform(x))
    print scaler.inverse_transform(kmeans.cluster_centers_)
    return (kmeans, scaler)


def classifyPlayers():
    client = MongoClient('localhost', 27017)
    db = client["pitchfx"]
    highVolumePower = []
    balancedPower = []
    defensive = []
    inPlay = []
    for player in db.players.find():
        if player.get('h_2014') == None:
            continue
        if player['hr_2014'] > 20:
            if player['avg_2014'] < 0.280:
                highVolumePower.append(player['player_id'])
            else:
                balancedPower.append(player['player_id'])
        elif player['avg_2014'] > 0.280 and player['ab_2014'] > 150:
            if float(player['bb_2014']) / (player['ab_2014'] + player['bb_2014']) > 0.10:
                defensive.append(player['player_id'])
            elif float(player['so_2014']) / (player['ab_2014'] + player['bb_2014']) < 0.16:
                inPlay.append(player['player_id'])
    print 'high vol = ', len(highVolumePower)
    print 'balanced = ', len(balancedPower)
    print 'defensive = ', len(defensive)
    print 'in play = ', len(inPlay)
    return {
        'highVol': highVolumePower,
        'balanced': balancedPower,
        'defensive': defensive,
        'inPlay': inPlay
    }

def classifyGeneralPlayer(player):
    powerScore = player['hr_2014'] / 20.0
    avgScore = player['avg_2014'] / 0.280
    if powerScore > (avgScore + 0.5):
        return 'highVol'
    elif powerScore > (avgScore - 0.5):
        return 'balanced'
    else:
        if float(player['so_2014']) / (player['ab_2014'] + player['bb_2014']) < 0.16:
            return 'inPlay'
        else:
            return 'defensive'


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


client = MongoClient('localhost', 27017)
db = client["pitchfx"]
num_clusters = 8
(kmeans, scaler) = classifyWithKmeans(8)
data = []
for i in range(num_clusters):
    data.append([])
for player in db.players.find():
    if player.get('h_2014') == None or player.get('h_2014') < 100:
        continue
    cluster = kmeans.predict(scaler.transform([kmeans_features(player)]))[0]
    data[cluster].append(player)
print data









