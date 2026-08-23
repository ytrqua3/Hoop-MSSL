<h1>Hoop MSSL</h1>
<h2>Introduction</h2>
"Hoop-MSSL: Multitask self-supervised representation learning on basketball spatiotemporal data" is an article that documents the structure of a transformer based model that learns about the game of basketball through movement data of individual players. This paper proposes a BERT-like encoder-decoder structured model which learns on 3 different tasks and proves that the encoder is highly generalizable to other decoders/tasks. 

<h2>Data</h2>
- a batch is a set of possessions
- each possession has trajectory of 11 agents (players and the ball)
- each trajectory is a sequence of 121 timesteps which stores (x, y, v: speed)
=> data = (N x 11 x 121 x 3) where N is the batch size

<h2>Model Structure</h2>
1. Augmentation: disorder augmentation and random masking (80%) to produce two views of each possession (32x11x121x3) -> (64x11x121x3)
2. Projection: project the 3 features to a larger dimension space of 128 (64x11x121x3) -> (64x11x121x128)
3. Positional Encoding
4. Temporal Transformer: explores the relationship between different timesteps within individual possessions
5. Spatial Transformer: explores the relationship between different agents within individual possessions
6. Decoders: calculates the loss of the model using three tasks
  i. Motion Reconstruction: MLP 128->3(x, y, v)
  ii. Player Role Identification: MLP 128 -> 2(offense, defense)
  iii. Contrastive learning: MLP 128 -> 128 
