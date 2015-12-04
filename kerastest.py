import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD

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
    feature_vec.append(1)
    feature_vec.append(pitch["break_angle"])
    feature_vec.append(pitch["break_length"])
    feature_vec.append(pitch["break_y"])
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


def train(model):
    client = MongoClient('localhost', 27017)
    db = client["pitchfx"]
    x = []
    y = []
    num_added = 0
    for pitch in db.pitches.find({"pitch_type":"FF"}, limit=100000):
        feature_vec = getFeaturesFF(pitch)
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
    model.fit(np.array(x), y, nb_epoch=20, batch_size=16)

def testTrain(model):
    client = MongoClient('localhost', 27017)
    db = client["pitchfx"]
    num_added = 0
    total_error = 0
    for pitch in db.pitches.find({"pitch_type":"FF"}, limit=100000):
        feature_vec = getFeaturesFF(pitch)
        if pitch["type"] == 'S':
            result = 1
        if pitch["type"] == 'B':
            continue
        if pitch["type"] == 'X':
            result = events[pitch["event"]]
        print model.predict(np.array([feature_vec]))[0]
        error = model.predict(np.array([feature_vec]))[0] - result
        #print 'prediction: ', reg.predict(scaler.transform([feature_vec]))[0], ' result: ', result 
        num_added += 1
        total_error += abs(error)
    print num_added, ' pitches tested'
    print 'Train Error = ', total_error, ' error per pitch = ', total_error/float(num_added)

def test(model):
    client = MongoClient('localhost', 27017)
    db = client["pitchfx"]
    num_added = 0
    total_error = 0
    for pitch in db.pitches.find({"pitch_type":"FF"}, limit=100000, skip=100000):
        feature_vec = getFeaturesFF(pitch)
        if pitch["type"] == 'S':
            result = 1
        if pitch["type"] == 'B':
            continue
        if pitch["type"] == 'X':
            result = events[pitch["event"]]
        error = model.predict(np.array([feature_vec]))[0] - result
        #print 'prediction: ', reg.predict(scaler.transform([feature_vec]))[0], ' result: ', result 
        num_added += 1
        total_error += abs(error)
    print num_added, ' pitches tested'
    print 'Error = ', total_error, ' error per pitch = ', total_error/float(num_added)


model = Sequential()

# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
# model.add(Dense(64, input_dim=10, init='uniform'))
# model.add(Activation('tanh'))
# model.add(Dropout(0.5))
# model.add(Dense(64, init='uniform'))
# model.add(Activation('tanh'))
# model.add(Dropout(0.5))
# model.add(Dense(1, init='uniform'))
# model.add(Activation('softmax'))
# error = .277

model.add(Dense(64, input_dim=13, init='uniform', activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(64, init='uniform', activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(1, init='uniform', activation='softmax'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)

train(model)
testTrain(model)
test(model)

# scaler = StandardScaler()
# reg = SGDRegressor(loss='squared_loss', n_iter=5)
# train(reg, scaler)
# testTrain(reg, scaler)
# test(reg, scaler)

# scaler = StandardScaler()
# gnb = GaussianNB()
# train(gnb, scaler)
# test(gnb, scaler)

# scaler = StandardScaler()
# svmreg = svm.SVR()
# train(svmreg, scaler)
# test(svmreg, scaler)

