"""Runner script to persistently run and convert data for new events."""
from MySQLdb import _mysql
import yaml
import subprocess
import os
import time
import datetime
import sqlite3

def connect_db():
    db_config_file = "chimefrb_db.yaml"
    with open(db_config_file) as f:
        db_config = yaml.safe_load(f)
    return _mysql.connect(**db_config)

def fetch_events(db, event_no):
    quert_str = (
        "SELECT baseband_raw_data.event_no, event_register.timestamp_utc "
        "FROM baseband_raw_data "
        "JOIN event_register ON baseband_raw_data.event_no=event_register.event_no "
        f"WHERE baseband_raw_data.event_no > {event_no};"
    )

    db.query(quert_str)
    r = db.store_result()
    events = []
    for _ in range(r.num_rows()):
        events.append(r.fetch_row()[0])
    return events

def convert_data(e):
    # If current time > event time + 1 hour OR Kotekan done with event
    # Look for files on the local area
    # If files found, make entry in datatrail, local file with local DB with state CONVERTING
    # multi-thread: conversion, md5sum and deletion of raw data of files
    # wait for all threads to finish
    # UPDATE local DB with state FINISHED
    print(e)


def connect_conversion_db():
    if not os.path.exists('bb_conversion.db'):
        con = sqlite3.connect('bb_conversion.db')
        sqlite = con.cursor()
        sqlite.execute('''CREATE TABLE conversion
                   (event_no int, status text)''')
    else:
        con = sqlite3.connect('bb_conversion.db')
        sqlite = con.cursor()
    return con, sqlite

def fetch_last_converted_event(sqlite):
    for row in sqlite.execute('SELECT * FROM conversion ORDER BY event_no DESC LIMIT 1'):
        event = row
    return event

def main():
    db = connect_db()
    conn, sqlite = connect_conversion_db()
    #sqlite.execute("INSERT INTO conversion VALUES (186052843, 'FINISHED')")
    #conn.commit()
    while True:
        last_event = fetch_last_converted_event(sqlite)
        print(last_event)
        events = fetch_events(db, last_event[0])
        for e in events:
            convert_data(e)
        time.sleep(3)
  
if __name__ == "__main__":
    main()
