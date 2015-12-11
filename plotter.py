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
import matplotlib.pyplot as plt
import pickle

# == test ==

saves = pickle.load(open("saves.p", "rb"))
reg = saves["reg"]
scaler = saves["scaler"]
vec = saves["vec"]

kershaw_saves = pickle.load(open("kershaw.p", "rb"))
avgs = kershaw_saves["avgs"]
sequence = kershaw_saves["sequence"]

def getLocation(zone):
    sz_top = 3.38
    sz_bot = 1.5
    if zone[0] == "T":
        pz = sz_bot + (sz_top - sz_bot)* 1/6
    elif zone[0] == "M":
        pz = sz_bot + (sz_top - sz_bot)* 3/6
    elif zone[0] == "B":
        pz = sz_bot + (sz_top - sz_bot)* 1/6
    else:
        print 'bad param to getLocation'
    if zone[1] == "L":
        px = -0.472
    elif zone[1] == "M":
        px = 0
    elif zone[1] == "R":
        px = 0.472
    else:
        print 'bad param to getLocation'
    return (px, pz)

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
    features["break_length/break_y"] = pitch["break_length"] / pitch["break_y"]
    features["break_length*start_speed"] = pitch["break_length"] * pitch["start_speed"]

    if pitch["pitch_type"] == "FF":
        features["FF"] = 1
    else:
        features["FF"] = 0
    if pitch["pitch_type"] == "CU":
        features["CU"] = 1
    else:
        features["CU"] = 0
    if pitch["pitch_type"] == "SL":
        features["SL"] = 1
    else:
        features["SL"] = 0
    if pitch["prev_type"] == "FF":
        features["prev_FF"] = 1
    else:
        features["prev_FF"] = 0
    if pitch["prev_type"] == "CU":
        features["prev_CU"] = 1
    else:
        features["prev_CU"] = 0
    if pitch["prev_type"] == "SL":
        features["prev_SL"] = 1
    else:
        features["prev_SL"] = 0

    attribs = ["start_speed", "start_speed^2", "px^2", "px", "pfx_x^2", "pfx_x", 
        "pfx_z^2", "pfx_z", "pz-mid", "pz-mid^2", "break_angle", "break_length", "break_y",
        "px*pz-mid", "px^2*(pz-mid)^2", "break_length/break_y", "break_length*start_speed"]
    for attrib in attribs:
        for pitch_type in ["prev_FF", "prev_CU", "prev_SL", "FF", "CU", "SL"]:
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
        features[zone1 + '_start_speed'] = features['start_speed']
        features[zone1 + 'break_length'] = features['break_length']
        features[zone1 + '_FF'] = features['FF']
        features[zone1 + '_SL'] = features['SL']
        features[zone1 + '_CU'] = features['CU']
        features[zone1 + '_prev=FF'] = features['prev_FF']
        features[zone1 + '_prev=SL'] = features['prev_SL']
        features[zone1 + '_prev=CU'] = features['prev_CU']
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

def color(num):
    val = (num - .45) * 4
    return [max(min(.8, 1-val),0), 0, min(val, 1)]

def plotpitches(prev_type, prev_zone, cluster, reg, scaler, vec):
    attribs = ["start_speed", "pfx_x", "pfx_z", "break_angle", "break_length", "break_y"]
    print '--------'
    print 'Previous pitch:', prev_type, prev_zone
    grids = []
    for pitch_type in ["FF", "SL", "CU"]:
        grid = []
        for ypos in ['T', 'M', 'B']:
            row = []
            for xpos in ['L', 'M', 'R']:
                pitch = {}
                zone = ypos + xpos
                (px, pz) = getLocation(zone)
                pitch['pitch_type'] = pitch_type
                pitch['px'] = px
                pitch['pz'] = pz
                pitch['sz_top'] = 3.38
                pitch['sz_bot'] = 1.5
                for attrib in attribs:
                    pitch[attrib] = avgs[pitch_type][attrib]
                pitch['prev_type'] = 'FF'
                (prev_px, prev_pz) = getLocation(prev_zone)
                pitch['prev_px'] = prev_px
                pitch['prev_pz'] = prev_pz
                prediction = reg.predict(scaler.transform(vec.transform(getFeatures(pitch, cluster)).toarray()))[0]
                row.append(color(prediction))
                print pitch_type, zone, ':', prediction
            grid.append(row)
        grids.append(grid)
    fig, axes = plt.subplots(1, 3, figsize=(8, 6),
                         subplot_kw={'xticks': [], 'yticks': []})

    fig.subplots_adjust(hspace=0.3, wspace=0.05)

    axes.flat[0].imshow(grid, interpolation='nearest')
    axes.flat[0].set_title("Fastball")
    axes.flat[1].imshow(grid, interpolation='nearest')
    axes.flat[1].set_title("Slider")
    axes.flat[2].imshow(grid, interpolation='nearest')
    axes.flat[2].set_title("Curveball")

    plt.plot()


def plotprev(pitch_type, location, cluster, avgs, reg, scaler, vec):
    data = {
        'FF':[],
        'SL':[],
        'CU':[]
    }
    attribs = ["start_speed", "pfx_x", "pfx_z", "break_angle", "break_length", "break_y"]
    for pitch_type in ["FF", "SL", "CU"]:
        grid = []
        for zone in ['TL', 'TM', 'TR', 'ML', 'MM', 'MR', 'BL', 'BM', 'BR']: 
            pitch = {}
            (px, pz) = getLocation(location)
            pitch['pitch_type'] = 'FF'
            pitch['px'] = px
            pitch['pz'] = pz
            pitch['sz_top'] = 3.38
            pitch['sz_bot'] = 1.5
            for attrib in attribs:
                pitch[attrib] = avgs[pitch_type][attrib]
            pitch['prev_type'] = pitch_type
            (prev_px, prev_pz) = getLocation(zone)
            pitch['prev_px'] = prev_px
            pitch['prev_pz'] = prev_pz
            prediction = reg.predict(scaler.transform(vec.transform(getFeatures(pitch, cluster)).toarray()))[0]
            data[pitch_type].append(prediction)

    N = 9
    pitches = (.7, .45, .5, .6, .7, .65, .55, .5, .62)

    ind = np.arange(N)  # the x locations for the groups
    width = 0.30       # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, data['FF'], width, color='r')
    rects2 = ax.bar(ind+width, data['SL'], width, color='b')
    rects3 = ax.bar(ind+width*2, data['CU'], width, color='g')

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Prediction')
    ax.set_xlabel('Zone of Previous Pitch')
    ax.set_title('Effectiveness based on previous pitch')
    ax.set_xticks(ind+width)
    ax.set_xticklabels( ('TL', 'TM', 'TR', 'ML', 'MM', 'MR', 'BL', 'BM', 'BR') )

    ax.legend( (rects1[0], rects2[0], rects3[0]), ('Fastball', 'Slider', 'Curveball'), loc=4 )

    def autolabel(rects):
        # attach some text labels
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x()+rect.get_width()/2., 1.05*height, '%.2f'% height,
                    ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    plt.ylim([0,.9])
    plt.plot()

def plotPreditionVsActual(pitch_type, location, graph_pitch_type, cluster, avgs, seq, reg, scaler, vec):
    attribs = ["start_speed", "pfx_x", "pfx_z", "break_angle", "break_length", "break_y"]
    key = pitch_type + ' ' + location
    actual = []
    predictions = []
    for zone in ['TL', 'TM', 'TR', 'ML', 'MM', 'MR', 'BL', 'BM', 'BR']:
        actual.append(seq[pitch_type + ' ' + location][graph_pitch_type + ' ' + zone])

        pitch = {}
        (px, pz) = getLocation(zone)
        pitch['pitch_type'] = graph_pitch_type
        pitch['px'] = px
        pitch['pz'] = pz
        pitch['sz_top'] = 3.38
        pitch['sz_bot'] = 1.5
        for attrib in attribs:
            pitch[attrib] = avgs[pitch_type][attrib]
        pitch['prev_type'] = pitch_type
        (prev_px, prev_pz) = getLocation(location)
        pitch['prev_px'] = prev_px
        pitch['prev_pz'] = prev_pz
        prediction = reg.predict(scaler.transform(vec.transform(getFeatures(pitch, cluster)).toarray()))[0]
        predictions.append(prediction)
    print actual
    print predictions

    N = 9

    ind = np.arange(N)  # the x locations for the groups
    width = 0.35       # the width of the bars

    fig, ax1 = plt.subplots()
    rects1 = ax1.bar(ind, predictions, width, color='r')
    ax2 = ax1.twinx()
    rects2 = ax2.bar(ind+width, actual, width, color='b')

    # add some text for labels, title and axes ticks
    ax1.set_ylabel('Predicted Effectiveness')
    ax1.set_ylim([.4,.8])
    ax1.set_xlabel('Pitch Position')
    ax1.set_title('Effectiveness Compared to Amount Thrown')
    ax1.set_xticks(ind+width)
    ax1.set_xticklabels( ('TL', 'TM', 'TR', 'ML', 'MM', 'MR', 'BL', 'BM', 'BR') )

    ax2.set_ylabel('Number Thrown')
    ax2.legend( (rects1[0], rects2[0]), ('Predicted Effectiveness', 'Number Thrown'), loc=4 )

    # def autolabel(rects):
    #     # attach some text labels
    #     for rect in rects:
    #         height = rect.get_height()
    #         ax.text(rect.get_x()+rect.get_width()/2., 1.05*height, '%.2f'% height,
    #                 ha='center', va='bottom')

    # autolabel(rects1)
    # autolabel(rects2)
    plt.plot()

plotpitches("FF", "BR", 6, reg, scaler, vec)
plotpitches("FF", "BR", 1, reg, scaler, vec)
plotprev('FF', 'BR', 6, avgs, reg, scaler, vec)
plotPreditionVsActual('FF', 'BR', 'FF', 6, avgs, sequence, reg, scaler, vec)


plt.show()
