from IPython.display import display


def detect_anomalies(df):
    print('--------------- detecting the anomalies ------------------')
    print('social media scren time is bigger than the spent time on screen') #37
    display((df['Social_Media_Hours'] > df['Screen_Time_Hours']).sum())
    print()
    print()
    print('---------- gpa that exceeds the limit --------------')
    display((df['GPA'] > 5).sum())
    print()
    print()
    print('---------------- sleep hours and screen time both exceed 24h -------------------')
    display((df['Sleep_Hours'] + df['Screen_Time_Hours'] > 24).sum())
    print()
    print()
    print('---------------- age and years of study do not match ------------------')
    min_age = 17
    res = df['Age'] < min_age + (df['Year_of_Study'] - 1).clip(lower=0)
    display(res.sum().sum())
