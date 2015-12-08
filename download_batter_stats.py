import json, yaml
import xml.etree.ElementTree as ET
from HTMLParser import HTMLParser
import httplib, urllib
import traceback
import re
from pymongo import MongoClient
from bson.objectid import ObjectId


def main():
    connection = httplib.HTTPConnection('gd2.mlb.com')
    connection.connect()
    client = MongoClient('localhost', 27017)
    db = client["pitchfx"]
    scrapeYear(connection, db, 2008)

def scrapeYear(connection, db, year):
    connection.request('GET', "/components/game/mlb/year_%04d/batters/" % (year))
    htmldata = connection.getresponse().read()
    battersParser = BattersHTMLParser()
    battersParser.feed(htmldata)
    batters = battersParser.batters
    for batter in batters:
        scrapeBatter(connection, db, year, batter)

def scrapeBatter(connection, db, year, batter):
    try:
        connection.request('GET', "/components/game/mlb/year_%04d/batters/%s.xml" % (year, batter))
        batterxml = connection.getresponse().read()
        root = ET.fromstring(batterxml)
        fields = {}
        fields["ab_%04d" % (year)] = int(root.attrib.get("s_ab", 0))
        fields["hr_%04d" % (year)] = int(root.attrib.get("s_hr", 0))
        fields["rbi_%04d" % (year)] = int(root.attrib.get("s_rbi", 0))
        fields["sb_%04d" % (year)] = int(root.attrib.get("s_sb", 0))
        fields["h_%04d" % (year)] = int(root.attrib.get("s_h", 0))
        fields["2b_%04d" % (year)] = int(root.attrib.get("s_double", 0))
        fields["3b_%04d" % (year)] = int(root.attrib.get("s_triple", 0))
        fields["1b_%04d" % (year)] = int(root.attrib.get("s_single", 0))
        fields["bb_%04d" % (year)] = int(root.attrib.get("s_bb", 0))
        fields["so_%04d" % (year)] = int(root.attrib.get("s_so", 0))   
        if fields["ab_%04d" % (year)] > 0:
            fields["avg_%04d" % (year)] = float(fields["h_%04d" % (year)]) / fields["ab_%04d" % (year)]
            fields["obp_%04d" % (year)] = float(fields["h_%04d" % (year)] + fields["bb_%04d" % (year)]) / (fields["ab_%04d" % (year)] + fields["bb_%04d" % (year)])
            fields["slg_%04d" % (year)] = float(fields["1b_%04d" % (year)] + 2*fields["2b_%04d" % (year)] + 3*fields["3b_%04d" % (year)] + 4*fields["hr_%04d" % (year)]) / fields["ab_%04d" % (year)]
        else:
            fields["avg_%04d" % (year)] = 0.0
            fields["obp_%04d" % (year)] = 0.0
            fields["slg_%04d" % (year)] = 0.0
        db.players.update_one(
            {"player_id": batter},
            {
                "$set": fields
            }
        )
        print 'updated player ', batter
    except:
        print 'failed on player ', batter

    

class BattersHTMLParser(HTMLParser):
    def __init__(self):
        HTMLParser.__init__(self)
        self.batters = []

    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            for attr in attrs:
                if attr[1].endswith('xml') and len(attr[1]) > 7:
                    self.batters.append(attr[1][:6])

if __name__ == "__main__":
    main()