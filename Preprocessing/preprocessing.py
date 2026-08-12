import json
import numpy as np
import io
import boto3
import py7zr
from scipy import interpolate
import sys
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from requests.exceptions import ReadTimeout
from nba_api.stats.endpoints import commonplayerinfo, playercareerstats, commonallplayers
import time
import random

def get_unique_moments(data):
  #get list of non replicating game clock, shot clock and moments per quarter
  prev_game_clocks = [[], [], [], []]
  unique_moments = [[], [], [], []]
  prev_shot_clocks = [[], [], [], []]

  for i, event in enumerate(data['events']):
    for moment in event['moments']:
      #[quarter, , game_clock, shot_clock, , 2d list of agent features]
      quarter = moment[0]
      game_clock = moment[2]
      shot_clock = moment[3]
      agents = moment[5]

      if quarter > 4:
        continue

      if (shot_clock is None):
        if game_clock > 24:
          continue
        else:
          shot_clock = game_clock
      if game_clock not in prev_game_clocks[quarter-1]:
        for agent in agents:
          agent.extend([game_clock, shot_clock, quarter])
        unique_moments[quarter-1].append(agents)
        prev_game_clocks[quarter-1].append(game_clock)
        prev_shot_clocks[quarter-1].append(shot_clock)

  return prev_game_clocks, prev_shot_clocks, unique_moments

def extract_possessions(prev_game_clocks, prev_shot_clocks, unique_moments):
    #record possessions (since logic to capture all possessions are too complicated, I only capture possessions that are 100% valid that start from shot clock=24s)
    possessions = [[], [], [], []]
    failed_sc = [[], [], [], []]
    failed_gc = [[], [], [], []]

    for quarter in range(4):
      gc = prev_game_clocks[quarter]
      sc = prev_shot_clocks[quarter]
      moments = unique_moments[quarter]
      quarter_possessions = possessions[quarter]
      possession = []
      in_possession = False #to find a new starting point

      #loop over the moments
      for i in range(1, len(gc)-1):

        if in_possession:

          #shot clock still going down and gc sc time decrease is consistent
          if (sc[i-1] > sc[i]) and (abs((gc[i-1] - gc[i]) - (sc[i-1] - sc[i])) < 0.1):
            possession.append(moments[i])

          #append possession when end of possession (shot clock resets to 24) or end of quarter
          elif ((sc[i-1] < sc[i]) and (sc[i] > 23.9)):
            if len(possession) > 10:
              quarter_possessions.append(possession)
            in_possession = False

          #in between possessions
          elif (sc[i] == 24.0) and ((sc[i+1] == 24.0) or sc[i+1] == gc[i+1]):
            continue

          #invalid possession
          else:
            failed_sc[quarter].append(sc[i])
            failed_gc[quarter].append(gc[i])
            possession = []
            in_possession = False

        #find the next possession's starting point (24s)
        else:

          if sc[i] == 24.0 and sc[i+1] == 24.0:
            continue

          #found the starting point
          elif (sc[i] > 23.9):
            in_possession = True
            possession = []

    return possessions

def process_possessions(possessions):
  possessions_arr = []
  for quarter_possessions in possessions:
    for i, possession in enumerate(quarter_possessions):

      try:
        possession = np.array(possession)
      except ValueError:
        quarter_possessions.pop(i)
        continue


      #add speed as feature
      added_possession = []
      for agent in np.transpose(possession, (1, 0, 2)):
        speed_col = np.insert(np.sqrt(np.diff(agent[:, 3])**2 + np.diff(agent[:, 4])**2) / np.abs(np.diff(agent[:, 5])), 0, 0).reshape(-1, 1)
        agent = np.hstack([agent, speed_col])
        added_possession.append(agent)
      possession = np.transpose(np.array(added_possession), (1, 0, 2))

      # ensure that a moment has 11 valid agents by masking and sort the agents by team_id
      valid_mask = []
      for j, moment in enumerate(possession):
        counts = {}
        for agent in moment:
          counts[agent[0]] = counts.get(agent[0], 0) + 1

        if len(counts.keys()) != 3:
          valid_mask.append(False)
          continue

        for k in counts.keys():
          if k == -1:
            if counts[k] != 1:
              valid_mask.append(False)
              break
          elif counts[k] != 5:
              valid_mask.append(False)
              break
        valid_mask.append(True)

        #sort the moment in ascending order of player_id
        possession[j] = moment[moment[:, 1].argsort()]
      possession = possession[np.array(valid_mask)]

      # if there are valid timesteps in the possession, continue
      if possession.shape[0] == 0:
        continue

      # the agent ball is always the first agent
      ball_moments = possession[:, 0, :]
      avg_x = ball_moments[:, 3].mean()

      #cut half court and flip the court so all possessions face the same way
      if avg_x > 47:
        idx = np.argmax(ball_moments[:, 3] > 47) #the moment that the ball crosses half court
        possession = possession[idx: ]
        ball_moments[:, 3] = 94 - ball_moments[:, 3]
        ball_moments[:, 4] = 50 - ball_moments[:, 4]

      else:
        idx = np.argmax(ball_moments[:, 3] <= 47)
        possession = possession[idx:]

      possessions_arr.append(possession)

  #remove rows where speed is nan or inf
  for i, possession in enumerate(possessions_arr):

    # Extract speed column
    speed = possession[:, :, -1] #[moments, agents, 1]

    # Check which timesteps have ALL valid speeds across agents
    # For a timestep to be valid, every agent's speed must be finite
    # Shape: (T,) - True if timestep is valid
    valid_timesteps = np.all(np.isfinite(speed), axis=1)

    # Keep only valid timesteps
    cleaned_possession = possession[valid_timesteps]
    if cleaned_possession.shape[0] != possession.shape[0]:
      possessions_arr[i] = cleaned_possession

  return possessions_arr

def interpolate_possessions(possessions_arr):
  interpolated_possessions = []
  for i, possession in enumerate(possessions_arr):
    original_timesteps = len(possession)
    target_len = 121

    # Create index arrays for interpolation
    old_indices = np.linspace(0, original_timesteps - 1, original_timesteps)
    new_indices = np.linspace(0, original_timesteps - 1, target_len)

    # Initialize output array
    interpolated_data = np.zeros((target_len, 11, 9))

    # Interpolate for each agent and each feature
    for agent in range(11):
        for feature in range(9):
            # Extract the feature trajectory for this agent
            trajectory = possession[:, agent, feature]

            # Create interpolation function (linear is typical for path data)
            f = interpolate.interp1d(old_indices, trajectory, kind='linear')

            # Apply to new indices
            interpolated_data[:, agent, feature] = f(new_indices)
    if interpolated_data.shape == (121, 11, 9):
      interpolated_possessions.append(interpolated_data)
  interpolated_possessions = np.array(interpolated_possessions)
  return interpolated_possessions

def append_player_info(interpolated_possessions, lookup_table):
  if interpolated_possessions.size == 0:
    return interpolated_possessions
  agent_ids = interpolated_possessions[:, :, 1:, 1].astype(int) #103, 121, 10
  # 103, 121, 10, 3 + 103, 121, 1, 3 -> 103, 121, 11, 3
  info = np.concatenate([lookup_table[agent_ids], np.zeros((interpolated_possessions.shape[0], interpolated_possessions.shape[1], 1, 3))], axis=2)
  final_possessions = np.concatenate([interpolated_possessions, info], axis=-1)
  return final_possessions

def json_to_npy(game_json_text, lookup_table):
  data = json.loads(game_json_text)

  try:
    prev_game_clocks, prev_shot_clocks, unique_moments = get_unique_moments(data)
  except Exception as e:
    print("❌ get_unique_moments function failed")
    print("Logging error: {e}")
    sys.stdout.flush()
    return np.array([])

  try:
    possessions = extract_possessions(prev_game_clocks, prev_shot_clocks, unique_moments)
  except Exception as e:
    print("❌ extract_possessions function failed")
    print("Logging error: {e}")
    sys.stdout.flush()
    return np.array([])

  try:
    possessions_arr = process_possessions(possessions)
  except Exception as e:
    print("❌ process_possessions function failed")
    print("Logging error: {e}")
    sys.stdout.flush()
    return np.array([])

  try:
    interpolated_possessions = interpolate_possessions(possessions_arr)
  except Exception as e:
    print("❌ interpolate_possessions function failed")
    print("Logging error: {e}")
    sys.stdout.flush()
    return np.array([])

  try:
    final_possessions = append_player_info(interpolated_possessions, lookup_table)
  except Exception as e:
    print("❌ append_player_info function failed")
    print("Logging error: {e}")
    return np.array([])

  return final_possessions

class NbaDataMemoryFactory(py7zr.io.WriterFactory):
  def __init__(self):
    self.buffers = {}

  def create(self, filename):
    buffer = io.BytesIO()
    self.buffers[filename] = buffer
    return buffer

def download_game(game_i, obj_info, ignore_fnames):
  s3_client = boto3.client('s3')
  bucket_name = 'hoop-mssl'

  #fetch the .7z file for the game
  s3_key = obj_info['Key']
  obj = s3_client.get_object(Bucket=bucket_name, Key=s3_key)['Body']
  file_stream = io.BytesIO(obj.read())

  factory = NbaDataMemoryFactory()
  game_json_text = None
  with py7zr.SevenZipFile(file_stream, mode='r') as z:
    for name in z.getnames():
      if name.endswith('.json'):
        if name in ignore_fnames:
          print(f"⏭️ game {name} is already processed")
          return game_i, "", name
        z.extract(targets=[name], factory=factory)
        buffer = factory.buffers[name]
        buffer.seek(0)
        game_json_text = buffer.read().decode('utf-8')
        return game_i, game_json_text, name
  print(f"❓ no json file found in {s3_key.split('/')[-1]}")
  sys.stdout.flush()
  return game_i, "", ""

def upload_processed_game(game_i, final_possessions, total_games, game_id):
  if final_possessions.size() == 0:
    print(f"❓ empty game {game_id}")
    return

  s3_client = boto3.client('s3')
  bucket_name = 'hoop-mssl'

  buffer = io.BytesIO()
  np.save(buffer, final_possessions)
  buffer.seek(0)

  if game_i < total_games * 0.7 :
      s3_key = f'processed_data/train/{game_id}.npy'
      print(f"✅ finished preprocessing game {game_id} with shape {final_possessions.shape} and saving to train")
      sys.stdout.flush()
  elif game_i > total_games * 0.85:
      s3_key = f'processed_data/test/{game_id}.npy'
      print(f"✅ finished preprocessing game {game_id} and saving to test")
      sys.stdout.flush()
  else:
      s3_key = f'processed_data/val/{game_id}.npy'
      print(f"✅ finished preprocessing game {game_id} and saving to val")
      sys.stdout.flush()
  s3_client.upload_fileobj(buffer, bucket_name, s3_key)

  buffer.close()

def main():
  bucket_name = 'hoop-mssl'
  data_prefix = 'nba_data/'
  lookup_table_path = '/content/lookup_table (1).npy'
  s3_client = boto3.client('s3')

  processed_file_response = s3_client.list_objects_v2(Bucket='hoop-mssl', Prefix='processed_data')['Contents']
  processed_fnames = [obj_info['Key'].split('/')[-1][:-4] + ".json" for obj_info in processed_file_response]

  response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=data_prefix)
  total_games = len(response['Contents'])

  print(f"🚀 extracting game data from 7z to json for {total_games} games")
  sys.stdout.flush()

  lookup_table = np.load(lookup_table_path)

  batch_size = 20
  for batch_start in range(0, total_games, batch_size):
    print(f"📦 Processing batch: games {batch_start} to {batch_start + batch_size - 1}")
    with ThreadPoolExecutor(max_workers=10) as thread_pool, ProcessPoolExecutor() as cpu_pool:
      download_futures = []
      for i, obj_info in enumerate(response['Contents'][batch_start : batch_start+20]):
        game_i = batch_start + i
        future = thread_pool.submit(download_game, game_i, obj_info, processed_fnames)
        download_futures.append(future)

      for download_future in download_futures:
        game_i, game_json_text, name = download_future.result()
        if game_json_text == "":
          continue
        game_id = name[:-5]
        cpu_future = cpu_pool.submit(json_to_npy, game_json_text, lookup_table)
        final_possessions = cpu_future.result()
        del game_json_text
        thread_pool.submit(upload_processed_game, game_i, final_possessions, total_games, game_id)


  print("✅ finished main")


main()
