from icalendar import Calendar
from pathlib import Path
import datetime

def get_events_from_ics(ics_file_path):
    events = []
    try:
        with open(ics_file_path, 'rb') as f:
            gcal = Calendar.from_ical(f.read())
        for component in gcal.walk():
            if component.name == "VEVENT":
                summary = component.get('summary')
                dtstart = component.get('dtstart').dt
                dtend = component.get('dtend').dt
                if isinstance(dtstart, datetime.datetime) and isinstance(dtend, datetime.datetime):
                     events.append({
                         "summary": str(summary),
                         "start": dtstart.isoformat(),
                         "end": dtend.isoformat(),
                         "description": str(component.get('description', '')),
                         "location": str(component.get('location', ''))
                     })
    except Exception as e:
        print(f"Error parsing {ics_file_path}: {e}")
    return events

