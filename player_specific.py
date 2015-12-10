from pymongo import MongoClient
import json

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


attribs = ["start_speed", "pfx_x", "pfx_z", "break_angle", "break_length", "break_y"]

for ptype in ["FF", "SL", "CU"]:
    for attrib in attribs:
        avgs[ptype][attrib] = 0
    avgs[ptype]["total"] = 0

for pitch in db.pitches.find({"pitcher": kershaw_id, "pitch_type":{"$in": ["FF", "SL", "CU"]}}):
    for attrib in attribs:
        avgs[pitch["pitch_type"]][attrib] += pitch[attrib]
    avgs[pitch["pitch_type"]]["total"] += 1

for ptype in ["FF", "SL", "CU"]:
    for attrib in attribs:
        avgs[ptype][attrib] = avgs[ptype][attrib] / float(avgs[ptype]["total"])

print json.dumps(avgs, indent=4)
