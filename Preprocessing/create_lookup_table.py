from requests.exceptions import ReadTimeout
import time
import random
from nba_api.stats.endpoints import commonplayerinfo, playercareerstats, commonallplayers
from tqdm import tqdm
import numpy as np

def create_players_info_lookup():
  season_roster = commonallplayers.CommonAllPlayers()
  roster_df = season_roster.get_data_frames()[0]
  roster_1516_df = roster_df[(roster_df['FROM_YEAR'].astype(int) <= 2016) & (roster_df['TO_YEAR'].astype(int) >= 2015)]
  player_ids = roster_1516_df['PERSON_ID'].to_list()
  max_key = max(player_ids)
  lookup_table = np.zeros((max_key+1, 3))


  HEADERS = {
    'Host': '://nba.com',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Origin': 'https://nba.com',
    'Referer': 'https://nba.com/'
  }

  position_table = {
      'guard': 0,
      'guard-forward': 1,
      'forward-guard': 2,
      'forward': 3,
      'forward-center': 4,
      'center-forward': 5,
      'center':7,
      'ball': -1
  }

  failed_ids = []
  print(len(player_ids))
  previous_failed_i = -2
  fail_streak = 0
  for i, id in enumerate(tqdm(player_ids)):

    id = int(id)
    try:
      player_data = commonplayerinfo.CommonPlayerInfo(player_id=id)
      info_df = player_data.get_data_frames()[0]
      height_str = info_df['HEIGHT'].values[0].split('-')
      if len(height_str) != 2:
        print(f"height problem id {id}, {height_str}")
        failed_ids.append(id)
        if i - previous_failed_i == 1:
          fail_streak += 1
          previous_failed_i = i
        if fail_streak >= 5:
          print("fail streak reached 5")
          break
        continue
      if height_str[1] == "":
        height_str = 0
      if height_str[0] == "":
        print(f"height_str[0] is empty id: {id}, {height_str}")
        failed_ids.append(id)
      height = int(height_str[0])*30.48 + int(height_str[1])*2.54
      position = position_table[info_df['POSITION'].values[0].lower()]

      try:
        stats = playercareerstats.PlayerCareerStats(player_id=id)
        all_star_1516 = bool((stats.get_data_frames()[2]['SEASON_ID'] == '2015-16').sum())
      except KeyError:
        all_star_1516 = False

      lookup_table[id] = [height, position, all_star_1516]

      time.sleep(random.uniform(1.5, 3.5))
    except (ReadTimeout) as e:
      print(f"sleeping for 30s due to timeout for player {id}")
      if i - previous_failed_i == 1:
        fail_streak += 1
        previous_failed_i = i
      if fail_streak >= 5:
        print("fail streak reached 5")
      failed_ids.append(id)

      time.sleep(30)

    except KeyError:
      print(f"id: {id}")
      failed_ids.append(id)
      if i - previous_failed_i == 1:
        fail_streak += 1
        previous_failed_i = i
      if fail_streak >= 5:
        print("fail streak reached 5")
      time.sleep(30)


  return lookup_table, failed_ids

lookup_table, failed_ids = create_players_info_lookup()
np.save('lookup_table.npy', lookup_table)
