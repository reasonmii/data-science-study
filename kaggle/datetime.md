df[col].dt.date         # YYYY-MM-DD(문자)
df[col].dt.year         # 연(4자리숫자)
df[col].dt.month        # 월(숫자)
df[col].dt.month_name() # 월(문자)

df[col].dt.day          # 일(숫자)
df[col].dt.time         # HH:MM:SS(문자)
df[col].dt.hour         # 시(숫자)
df[col].dt.minute       # 분(숫자)
df[col].dt.second       # 초(숫자)

df[col].dt.quarter       # 분기(숫자)
df[col].dt.weekday_name  # 요일이름(문자) (=day_name())
df[col].dt.weekday       # 요일숫자(0-월, 1-화) (=dayofweek)
df[col].dt.weekofyear    # 연 기준 몇주째(숫자) (=week)
df[col].dt.dayofyear     # 연 기준 몇일째(숫자)
df[col].dt.days_in_month # 월 일수(숫자) (=daysinmonth)

df[col].dt.is_leap_year     # 윤년 여부
df[col].dt.is_month_start   # 월 시작일 여부
df[col].dt.is_month_end     # 월 마지막일 여부
df[col].dt.is_quarter_start # 분기 시작일 여부
df[col].dt.is_quarter_end   # 분기 마지막일 여부
df[col].dt.is_year_start    # 연 시작일 여부
df[col].dt.is_year_end      # 연 마지막일 여부
