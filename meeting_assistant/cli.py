from __future__ import annotations
import argparse, json
from .db import MeetingDB
from .logging import EventLogger
from .tools import MeetingTools
from .web import serve

def main():
    parser=argparse.ArgumentParser(prog="meeting-assistant"); parser.add_argument("--db",default="data/meetings.sqlite3"); sub=parser.add_subparsers(dest="command",required=True)
    t=sub.add_parser("transcribe"); t.add_argument("audio"); t.add_argument("--title")
    w=sub.add_parser("web"); w.add_argument("--host",default="0.0.0.0"); w.add_argument("--port",type=int,default=8080)
    e=sub.add_parser("export"); e.add_argument("meeting_id")
    args=parser.parse_args(); logger=EventLogger(); tools=MeetingTools(MeetingDB(args.db),logger)
    if args.command=="transcribe": print(tools.transcribe_meeting(args.audio,args.title))
    elif args.command=="export": print(json.dumps(tools.export_minutes(args.meeting_id),ensure_ascii=False,indent=2))
    else: serve(tools,args.host,args.port)
if __name__ == "__main__": main()
