import unicodecsv
def csvread(filein):
    with open(filein,'rb') as f:
        reader = unicodecsv.DictReader(f)
        return list(reader)

enrollments = csvread('csv/enrollments.csv')
daily_engagement = csvread('csv/daily_engagement.csv')
project_submissions = csvread('csv/project_submissions.csv')

from datetime import datetime as dt

def parse_date(date):
    return None if date == '' else dt.strptime(date, '%Y-%m-%d')

def parse_maybe_int(i):
    return None if i == '' else int(i)

for x in enrollments:
    x['cancel_date'] = parse_date(x['cancel_date'])
    x['days_to_cancel'] = parse_maybe_int(x['days_to_cancel'])
    x['is_canceled'] = x['is_canceled'] == 'True'
    x['is_udacity'] = x['is_udacity'] == 'True'
    x['join_date'] = parse_date(x['join_date'])

for x in daily_engagement:
    x['lessons_completed'] = int(float(x['lessons_completed']))
    x['num_courses_visited'] = int(float(x['num_courses_visited']))
    x['projects_completed'] = int(float(x['projects_completed']))
    x['total_minutes_visited'] = float(x['total_minutes_visited'])
    x['utc_date'] = parse_date(x['utc_date'])

for x in project_submissions:
    x['completion_date'] = parse_date(x['completion_date'])
    x['creation_date'] = parse_date(x['creation_date'])

def num_rows(y):
    accum = 0
    for x in y[0]:
        accum += 1
    return accum

def unique_students(y):
    ids = set([])
    accum = 0
    for x in y:
        if x['account_key'] not in ids:
            ids.add(x['account_key'])
            accum += 1
    return accum

enrollment_num_rows = num_rows(enrollment)
enrollment_num_unique_students = unique_students(enrollment)

engagement_num_rows = num_rows(daily_engagement)
engagement_num_unique_students = unique_students(daily_engagement)

submission_num_rows = num_rows(project_submissions)
submission_num_unique_students = unique_students(project_submissions)
