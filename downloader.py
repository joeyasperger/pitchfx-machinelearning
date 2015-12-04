import json, yaml
import xml.etree.ElementTree as ET
from HTMLParser import HTMLParser
import httplib, urllib
import traceback
import re
from pymongo import MongoClient
from bson.objectid import ObjectId

pitch_attribs = [
    {'name': 'des', 'type': 'str'},
    {'name': 'type', 'type': 'str'}, 
    {'name': 'x', 'type': 'float'}, 
    {'name': 'y', 'type': 'float'}, 
    {'name': 'start_speed', 'type': 'float'}, 
    {'name': 'end_speed', 'type': 'float'}, 
    {'name': 'sz_top', 'type': 'float'}, 
    {'name': 'sz_bot', 'type': 'float'}, 
    {'name': 'pfx_x', 'type': 'float'}, 
    {'name': 'pfx_z', 'type': 'float'}, 
    {'name': 'px', 'type': 'float'}, 
    {'name': 'pz', 'type': 'float'},
    {'name': 'x0', 'type': 'float'},
    {'name': 'y0', 'type': 'float'}, 
    {'name': 'z0', 'type': 'float'}, 
    {'name': 'vx0', 'type': 'float'}, 
    {'name': 'vy0', 'type': 'float'}, 
    {'name': 'vz0', 'type': 'float'}, 
    {'name': 'ax', 'type': 'float'}, 
    {'name': 'ay', 'type': 'float'}, 
    {'name': 'az', 'type': 'float'}, 
    {'name': 'break_y', 'type': 'float'}, 
    {'name': 'break_angle', 'type': 'float'}, 
    {'name': 'break_length', 'type': 'float'},
    {'name': 'pitch_type',  'type': 'str'}, 
    {'name': 'type_confidence', 'type': 'float'}, 
    {'name': 'zone', 'type': 'int'}, 
    {'name': 'nasty', 'type': 'int'}, 
    {'name': 'spin_dir', 'type': 'float'}, 
    {'name': 'spin_rate', 'type': 'float'}
]


def main():
    connection = httplib.HTTPConnection('gd2.mlb.com')
    connection.connect()
    client = MongoClient('localhost', 27017)
    db = client["pitchfx"]
    for year in range(2008, 2016):
        scrapeYear(connection, db, year)

def scrapeYear(connection,db , year):
    for month in range(4,12):
        connection.request('GET', "/components/game/mlb/year_%04d/month_%02d/" % (year, month))
        htmldata = connection.getresponse().read()
        monthParser = MonthHTMLParser()
        monthParser.feed(htmldata)
        days = monthParser.days
        for day in days:
            dayInt = int(re.sub('[^0-9]','', day))
            scrapeDay(connection, db, year, month, dayInt)


def scrapeDay(connection, db, year, month, day):
    connection.request('GET', "/components/game/mlb/year_%04d/month_%02d/day_%02d/" % (year, month,day))
    dayhtml = connection.getresponse().read()
    dayParser = DayHTMLParser()
    dayParser.feed(dayhtml)
    games = dayParser.games
    for game in games:
        scrapeGame(connection, db, year, month, day, game)

def addPlayersToDB(db, root):
    for team in root:
        for person in team:
            if person.tag != 'player':
                continue
            player = person
            team = player.attrib.get('team_abbrev')
            if team == None:
                team = player.attrib.get('parent_team_abbrev')
            playerid = player.attrib["id"]
            if db.players.find_one({"player_id": playerid}) != None:
                continue # already in db
            player_json = {
                'first_name': player.attrib['first'],
                'last_name': player.attrib['last'],
                'throws': player.attrib['rl'],
                'bats': player.attrib.get('bats'),
                'position': player.attrib['position'],
                'team': team,
                'player_id': playerid
            }
            db.players.insert_one(player_json)


def scrapeGame(connection, db, year, month, day, game):
    connection.request('GET', "/components/game/mlb/year_%04d/month_%02d/day_%02d/%sinning/inning_all.xml" % (year, month, day, game))
    inningxml = connection.getresponse().read()
    try:
        root = ET.fromstring(inningxml)
    except:
        print 'PARSE ERROR'
        print '-----'
        print inningxml
        print '-----'
        print "/components/game/mlb/year_%04d/month_%02d/day_%02d/%sinning/inning_all.xml" % (year, month, day, game)
        error = {
            'xml': inningxml,
            'url': "/components/game/mlb/year_%04d/month_%02d/day_%02d/%sinning/inning_all.xml" % (year, month, day, game)
        }
        #global errors
        #errors.append(error)
        return
    connection.request('GET', "/components/game/mlb/year_%04d/month_%02d/day_%02d/%splayers.xml" % (year, month, day, game))
    playerxml = connection.getresponse().read()
    # try:
    addPlayersToDB(db, ET.fromstring(playerxml))
    # except:
    #     print 'PLAYER ERROR'
    #     print '-----'
    #     print inningxml
    #     print '-----'
    #     print "/components/game/mlb/year_%04d/month_%02d/day_%02d/%splayer.xml" % (year, month, day, game)
    score = {'home': 0, 'away': 0}
    pitches = db.pitches
    pitch_num = 0
    for inning in root:
        for halfInning in inning:
            runners = {'1B': 0, '2B': 0, '3B': 0}
            outs = 0
            for inning_event in halfInning:
                # inning_event can be atbat or action (don't need to do anything for actions)
                if inning_event.tag == 'atbat':
                    atbat = inning_event
                    count = [0,0]
                    pitcher = atbat.attrib["pitcher"]
                    batter = atbat.attrib["batter"]
                    pitcher_hand = atbat.attrib["p_throws"]
                    batter_hand = atbat.attrib["stand"]
                    for event in atbat:
                        # event can be pitch, po, or runner
                        if event.tag == 'pitch':
                            pitch = event
                            pitch_num += 1
                            pitch_json = {}
                            for attr in pitch_attribs:
                                val = pitch.attrib.get(attr["name"])
                                if val != None and val != "":
                                    if attr["type"] == "float":
                                        val = float(val)
                                    elif attr["type"] == "int":
                                        val = int(val)
                                    pitch_json[attr["name"]] = val
                            pitch_json['balls'] = count[0]
                            pitch_json['strikes'] = count[1]
                            pitch_json['outs'] = outs
                            if pitch.attrib['type'] == 'X':
                                pitch_json['event'] = atbat.attrib['event']
                            # TO ADD: score, baserunners
                            if pitch.attrib['type'] == 'S' and count[1] < 3:
                                count[1] += 1
                            elif pitch.attrib['type'] == 'B':
                                count[0] += 1
                            pitch_json["pitcher"] = pitcher
                            pitch_json["batter"] = batter
                            pitch_json["pitcher_hand"] = pitcher_hand
                            pitch_json["batter_hand"] = batter_hand
                            pitch_json["game"] = game
                            pitch_json["year"] = year
                            pitch_json["month"] = month
                            pitch_json["day"] = day
                            pitch_json["atbat_num"] = atbat.attrib["num"]
                            pitch_json["pitch_num"] = pitch_num
                            pitch_json["inning"] = inning.attrib["num"]
                            pitch_json["half"] = halfInning.tag
                            if atbat.get('score') == "T":
                                if halfInning.tag == 'top':
                                    pitch_json["scored"] = int(atbat.attrib["away_team_runs"]) - score["away"]
                                else:
                                    pitch_json["scored"] = int(atbat.attrib["home_team_runs"]) - score["home"]
                            pitch_json['1B'] = runners['1B']
                            pitch_json['2B'] = runners['2B']
                            pitch_json['3B'] = runners['3B']                            
                            pitch_id = pitches.insert_one(pitch_json).inserted_id

                        if event.tag == 'runner':
                            runner = event
                            if runner.attrib['start'] != "":
                                runners[runner.attrib["start"]] -= 1
                            if runner.attrib['end'] != "":
                                runners[runner.attrib["end"]] += 1
                            if runner.attrib.get('score') == 'T':
                                if halfInning.tag == 'top':
                                    score['away'] += 1
                                else:
                                    score['home'] += 1
                    # end of atbat
                    if atbat.get('score') == "T":
                        assert(score['home'] == int(atbat.attrib["home_team_runs"]))
                        assert(score['away'] == int(atbat.attrib["away_team_runs"]))
    print game


class MonthHTMLParser(HTMLParser):
    def __init__(self):
        HTMLParser.__init__(self)
        self.days = []

    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            for attr in attrs:
                if attr[1].startswith('day'):
                    self.days.append(attr[1])

class DayHTMLParser(HTMLParser):
    def __init__(self):
        HTMLParser.__init__(self)
        self.games = []

    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            for attr in attrs:
                if attr[1].startswith('gid'):
                    self.games.append(attr[1])

if __name__=="__main__":
    main()