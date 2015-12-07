from pymongo import MongoClient

def main():
    count = 0
    client = MongoClient('localhost', 27017)
    db = client["pitchfx"]
    for pitch in db.pitches.find():
        count += 1
        previous_pitch = db.pitches.find_one({"game":pitch["game"], "atbat_num":pitch["atbat_num"], "pitch_num": pitch["pitch_num"] - 1})
        if previous_pitch != None:
            db.pitches.update({
                '_id':pitch['_id']
            },{
                '$set': {
                    "prev_type": previous_pitch.get("pitch_type"),
                    'prev_pz': previous_pitch.get("pz"),
                    'prev_px': previous_pitch.get("px")
                }
            })
        else:
            db.pitches.update({
                '_id':pitch['_id']
            },{
                '$set': {
                    "prev_type": None,
                    'prev_pz': None,
                    'prev_px': None
                }
            })
        print count, ': ', pitch['_id']



if __name__ == "__main__":
    main()