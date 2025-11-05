import nflreadpy as nfl
import pandas as pd
import matplotlib.pyplot as plt

def jaxson_dart_details(plotGraph = False, export = False, printInfo = False):
    # Loads data from most recent season
    pbp_2025 = nfl.load_pbp(2025).to_pandas()

    # Drops rows indicating end of play (quarter end, two minute warning, end of OT)
    pbp_2025 = pbp_2025.dropna(subset=['play_type'])

    # Selects only passing plays
    passes_25 = pbp_2025[pbp_2025['play_type'] == 'pass']

    # Selects only passes by Jaxson Dart
    dart_passes_25 = passes_25.query("passer_player_name == 'J.Dart'").copy()

    # Records the amount of sacks taken
    sacks = len(dart_passes_25.query("sack == 1"))

    # Creates a new column with value 1 if the pass happened when the expected pass prob was less than 0.5, 0 otherwise
    dart_passes_25['run_leaning_pass'] = dart_passes_25.apply(
        lambda x: 1 if x['xpass'] <= 0.50 else 0, axis=1
    )
    play_action_proxy_count = dart_passes_25['run_leaning_pass'].sum()

    # Two point conversion data is sparse and does not have pass location info
    # Sacks should not be counted for total pass yardage
    dart_passes_25 = dart_passes_25.query("sack != 1 and extra_point_attempt != 1 and two_point_attempt != 1")

    if printInfo:
        total_passing_yards = dart_passes_25['yards_gained'].sum()
        total_passes = len(dart_passes_25)
        
        if total_passes > 0:
            incomplete_count = len(dart_passes_25.query("incomplete_pass == 1"))
            complete_count = len(dart_passes_25.query("incomplete_pass == 0"))
            completion_ratio = complete_count / total_passes

            passes_to_left = len(dart_passes_25.query("pass_location == 'left'"))
            passes_to_mid = len(dart_passes_25.query("pass_location == 'middle'"))
            passes_to_right = len(dart_passes_25.query("pass_location == 'right'"))

            print(f"J. Dart has thrown {play_action_proxy_count} passes in run-leaning situations.")

            print(f"J. Dart has thrown for {total_passing_yards} yards this season over {total_passes} attempts, at an average of {total_passing_yards / total_passes:.2f} yards per throw.\n")
            print(f"J. Dart has thrown {incomplete_count} incomplete passes and has a completion percentage of {completion_ratio * 100:.2f}%.")
            print(f"Excluding the incomplete passes, J. Dart has an average yards per completion of {total_passing_yards / complete_count:.2f} yards.")

            print(f"J. Dart has thrown passes to the left {passes_to_left} times, or on {passes_to_left / total_passes * 100:.2f}% of snaps.")
            print(f"J. Dart has thrown passes to the middle {passes_to_mid} times, or on {passes_to_mid / total_passes * 100:.2f}% of snaps.")
            print(f"J. Dart has thrown passes to the right {passes_to_right} times, or on {passes_to_right / total_passes * 100:.2f}% of snaps.")

            print(f"J. Dart has been sacked {sacks} times, or on {sacks / (total_passes + sacks) * 100:.2f}% of passing snaps.")

    # Creates a new column labeled 'air_yard_category' with the air yards of the pass in bins
    bins = [-100, 0, 10, 20, 100]
    labels = ['Behind LOS', 'Short [0-10)', 'Intermediate [10-20)', 'Deep [20+)']
    dart_passes_25['air_yard_category'] = pd.cut(
        dart_passes_25['air_yards'], 
        bins=bins, 
        labels=labels, 
        right=False,
        include_lowest=True
    )

    if plotGraph:
        chart_data = dart_passes_25.groupby(['pass_location', 'air_yard_category'], observed=False).size().unstack(fill_value=0)

        fig, ax = plt.subplots(figsize=(10, 6))
        chart_data.plot(kind='bar', stacked=True, ax=ax, rot=0)
        ax.set_title(f"J. Dart (NYG) Pass Distribution by Location & Distance (Total Throws: {dart_passes_25.shape[0]})", 
                fontsize=14)
        ax.set_xlabel("Pass Location", fontsize=12)
        ax.set_ylabel("Number of Throws", fontsize=12)
        ax.legend(title='Air Yard Distance', bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.show()

    # Counts the number of passes per week
    weekly_counts = dart_passes_25.groupby('week')['play_id'].transform('count')
    dart_passes_25['weekly_counts'] = weekly_counts

    # Reduces the size and dimensionality of the dataset
    columns_to_keep = [
        'play_id', 'game_id', 'week', 'desc', 'posteam', 'defteam', 'down', 'ydstogo',
        'yardline_100', 'qtr', 'game_seconds_remaining', 'score_differential', 'play_type',
        'passer_player_name', 'receiver_player_name', 'pass_location', 'air_yards', 
        'yards_after_catch', 'passing_yards', 'complete_pass', 'incomplete_pass', 'sack',
        'interception', 'touchdown', 'first_down_pass', 'epa', 'cpoe', 'xpass',
        'air_yard_category', 'weekly_counts'
    ]
    dart_passes_25 = dart_passes_25[columns_to_keep]

    if export:
        dart_passes_25.to_csv('dart_passing_analysis.csv', index=False)

if __name__ == "__main__":
    jaxson_dart_details(plotGraph=False, export=True, printInfo=True)