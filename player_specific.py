from pymongo import MongoClient
import json
import pickle

client = MongoClient('localhost', 27017)
db = client["pitchfx"]

kershaw_id = "477132"

avgs = {
    "FF":{},
    "SL":{},
    "CU":{}
}

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

sequence = {}
for ptype in ["FF", "SL", "CU"]:
    for zone in ['TL', 'TM', 'TR', 'ML', 'MM', 'MR', 'BL', 'BM', 'BR']:
        sequence[ptype + ' ' + zone] = {}

attribs = ["start_speed", "pfx_x", "pfx_z", "break_angle", "break_length", "break_y"]

for ptype in ["FF", "SL", "CU"]:
    for attrib in attribs:
        avgs[ptype][attrib] = 0
    avgs[ptype]["total"] = 0

for pitch in db.pitches.find({"pitcher": kershaw_id, "pitch_type":{"$in": ["FF", "SL", "CU"]}}):
    for attrib in attribs:
        avgs[pitch["pitch_type"]][attrib] += pitch[attrib]
    avgs[pitch["pitch_type"]]["total"] += 1
    if pitch['prev_type'] != None and pitch['prev_type'] in ["FF", "SL", "CU"] and pitch['pitch_type'] in ["FF", "SL", "CU"]:
        prevkey = pitch['prev_type'] + ' ' + getZone(pitch['prev_px'], pitch['prev_pz'], pitch['sz_top'], pitch['sz_bot'])
        currentkey = pitch['pitch_type'] + ' ' + getZone(pitch['px'], pitch['pz'], pitch['sz_top'], pitch['sz_bot'])
        sequence[prevkey][currentkey] = sequence[prevkey].get(currentkey, 0) + 1
print json.dumps(sequence, indent=4, sort_keys=True)

for ptype in ["FF", "SL", "CU"]:
    for attrib in attribs:
        avgs[ptype][attrib] = avgs[ptype][attrib] / float(avgs[ptype]["total"])

print json.dumps(avgs, indent=4)

saves = {
    'sequence': sequence,
    'avgs': avgs
}

pickle.dump(saves, open('kershaw.p', 'wb'))
