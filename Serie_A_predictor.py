import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


'''idee per future implementazioni:
visualizzazioni:
-heatmap delle correlazioni tra le variabili'''


'''
Date = data della partita
Time = ora della partita
Comp = competizione
Round = giornata
Day = giorno della settimana
Venue = in casa o fuor casa
Result = risultato finale
GF = goal fatti
GA = goal subiti
Opponent = avversario
xG = expected goals
xGA = expected goals against
Poss = possesso palla
Attendance = numero di spettatori
Captain = capitano della squadra
Formation = formazione iniziale
Opp formation = formazione iniziale avversario
Referee = arbitro
Match Report = link al report della partita
Notes = note sulla partita
Sh = tiri totali
SoT = tiri in porta
Dist = distanza dalla porta media per tiro (yards)
FK = calci di punizione
PK = calci di rigore segnati
PKatt = rigori tentati
Season = stagione
Team = squadra
'''

file_name = 'matches_serie_A.csv'
df = pd.read_csv(file_name)

def data_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    
    # Select relevant columns and create or changing new ones
    df = df[['Date', 'Round', 'Venue', 'Result', 'GF', 'GA', 'Team' ,'Opponent', 'xG', 'xGA', 'Poss', 'Sh', 'SoT', 'Dist', 'FK', 'PK', 'PKatt', 'Season']].copy() 
    df['GD'] = df['GF'] - df['GA'] #Goal Difference
    df['Season'] = df['Season'].apply(lambda x: f"{(int(x)-1)}-{x}") #Change season format from '2024' to '2023-2024'

    # Handle missing values 
    # Only four rows have missing values as seen in df.info(). In two of these rows, the 'Dist' value is missing because there were no shots taken ('Sh' = 0).
    df.loc[[2979, 3705], 'Dist'] = 0.0 #Set Dist to 0 where Sh = 0
    df.fillna({'xG': df['xG'].mean(), 'xGA': df['xGA'].mean(), 'Dist': df['Dist'].mean(), 'FK': df['FK'].mean()}, inplace=True)
    
    # Change data types
    for i in ['FK', 'Poss', 'Sh' ,'SoT']:
        df[i] = df[i].astype(int)
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    
    #df_cleaned = df.to_csv('matches_serie_A_cleaned.csv', index=False)

    return df


def create_summary(df: pd.DataFrame) -> pd.DataFrame:

    # Create a summary dataframe with aggregated statistics for each team per season
    summary = df.groupby(['Season', 'Team']).agg(
        Matches_Played=('Result', 'count'),
        Wins=('Result', lambda x: (x == 'W').sum()),
        Draws=('Result', lambda x: (x == 'D').sum()),
        Losses=('Result', lambda x: (x == 'L').sum()),
        Points=('Result', lambda x: (x == 'W').sum() * 3 + (x == 'D').sum()),
        Goals_For=('GF', 'sum'),
        Goals_Against=('GA', 'sum'),
        Goal_Difference=('GD', 'sum'),
        Total_xG=('xG', 'sum'),
        Total_xGA=('xGA', 'sum'),
        Average_Possession=('Poss', 'mean'),
        Total_Shots=('Sh', 'sum'),
        Total_Shots_on_Target=('SoT', 'sum'),
        Average_Distance_per_Shot=('Dist', 'mean'),
        Total_FK=('FK', 'sum'),
        Total_PK_Attempted=('PKatt', 'sum'),
        Total_PK_Scored=('PK', 'sum')
    ).sort_values(['Season', 'Points'], ascending=[True, False]).reset_index()

    # Calculate win, draw, and loss percentages
    summary['Win_Percentage'] = ((summary['Wins'] / summary['Matches_Played']) * 100)
    summary['Draw_Percentage'] = ((summary['Draws'] / summary['Matches_Played']) * 100)
    summary['Loss_Percentage'] = ((summary['Losses'] / summary['Matches_Played']) * 100)
    summary['Position'] = summary.index % 20 + 1

    # Round float columns to two decimal places
    columns = summary.columns.tolist()
    for c in columns:
        if summary[c].dtype in ['float64']:
            summary[c] = summary[c].round(2)
        else:
            pass

    # Save the summary dataframe to a CSV file
    # matches_series_A_summary = summary.to_csv('summary_serie_A.csv', index=False)

    return summary



def promoted_teams(df: pd.DataFrame) -> pd.DataFrame:
    
    promoted = ['Sassuolo', 'Pisa', 'Cremonese']
    promoted_data = {}
    for i in promoted: 
        team_rows = df.loc[df['Team'] == i]
        if i in df['Team'].tolist():
            latest_season = team_rows.sort_values('Season').iloc[-1] # I consider only the latest season. Much more realistic.
            promoted_data[i] = pd.DataFrame([latest_season])
        else:
            # I don't have data for 'Pisa'. I assume the data as the means of the other two promoted teams.
            mean = pd.concat([
                df.loc[df['Team'] == 'Sassuolo'].sort_values('Season').iloc[[-1]],
                df.loc[df['Team'] == 'Cremonese'].sort_values('Season').iloc[[-1]]
            ]).mean(numeric_only=True)

            empty_row = {col: 0 for col in df.columns}
            empty_row['Team'] = i
            empty_row['Season'] = ''

            for col in mean.index:
                empty_row[col] = mean[col]
            promoted_data[i] = pd.DataFrame([empty_row])


    data = pd.concat(list(promoted_data.values()), ignore_index = True)
    data = data.drop(columns=['Season'])

    return data

def split_summary_by_season(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:

    # Split the summary dataframe into separate CSV files for each season
    season = df['Season'].unique().tolist()
    
    # Create a separate CSV file for each season
    for s in season:
        summary_per_season = df[df['Season'] == s]
        summary_per_season.to_csv(f'Season data/summary_{s}.csv', index=False)

    # Create train and test sets, using the last season as the test set
    test_season = df[df['Season'] == season[-1]]
    test_season = test_season.drop(columns = ['Season'])

    # Drop relegated teams and append promoted ones
    test_season = test_season.iloc[:17, :]
    promoted = promoted_teams(df)
    test_season = pd.concat([test_season, promoted], axis= 0, ignore_index=True)
    test_season = test_season.set_index('Team')

    train_seasons = pd.concat([df[df['Season'] == s] for s in season if s != season[-1]], ignore_index=True)
    train_seasons = train_seasons.drop(columns= ['Season'])
    train_seasons = train_seasons.set_index('Team')
    train_target = pd.Series(train_seasons['Position'])


    #test_season.to_csv(f'test_season_{season[-1]}.csv')
    #train_seasons.to_csv('train_seasons.csv')

    return train_seasons, train_target, test_season


def build_and_train_model(X: pd.DataFrame, y: pd.Series) -> Pipeline:

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=42,
            class_weight="balanced"
        ))
    ])
    model.fit(X, y)
    return model



def predict_league_table(model: Pipeline, features: pd.DataFrame) -> pd.DataFrame:
    
    probas = model.predict_proba(features)
    classes = model.named_steps['rf'].classes_
    exp_positions = probas.dot(classes)
    prediction_df = pd.DataFrame({
        "team": features.index,
        "expected_position": exp_positions
    })
    prediction_df['predicted_rank'] = prediction_df.index + 1
    return prediction_df


def main():

    data_cleaned = data_cleaning(df)
    summary = create_summary(data_cleaned)
    X_train, y_train, X_test = split_summary_by_season(summary)
    model = build_and_train_model(X_train, y_train)
    predictions = predict_league_table(model, X_test)

    print("Predicted Serie A 2025/26 table:")
    for _, row in predictions.iterrows():
        print(
            f"{int(row['predicted_rank'])}. {row['team']} "
            f"(expected pos {row['expected_position']:.2f})"
        )


if __name__ == '__main__':
    main()

'''
Result:
Predicted Serie A 2025/26 table:
1. Napoli (expected pos 2.62)
2. Internazionale (expected pos 2.37)
3. Atalanta (expected pos 3.25)
4. Juventus (expected pos 4.06)
5. Roma (expected pos 4.58)
6. Lazio (expected pos 5.81)
7. Milan (expected pos 6.18)
8. Bologna (expected pos 7.39)
9. Fiorentina (expected pos 6.81)
10. Como (expected pos 10.57)
11. Torino (expected pos 11.80)
12. Udinese (expected pos 13.00)
13. Genoa (expected pos 12.95)
14. Hellas Verona (expected pos 15.49)
15. Cagliari (expected pos 15.14)
16. Parma (expected pos 15.65)
17. Lecce (expected pos 16.77)
18. Sassuolo (expected pos 18.55)
19. Pisa (expected pos 18.68)
20. Cremonese (expected pos 18.69)
'''