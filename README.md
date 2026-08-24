<h1>Hoop MSSL</h1>
<h2>Introduction</h2>
"Hoop-MSSL: Multitask self-supervised representation learning on basketball spatiotemporal data" is an article that documents the structure of a transformer based model that learns about the game of basketball through movement data of individual players. This paper proposes a BERT-like encoder-decoder structured model which learns on 3 different tasks and proves that the encoder is highly generalizable to other decoders/tasks. 

<h2>Data</h2>
<ul>
<li>a batch is a set of possessions 
<li>each possession has trajectory of 11 agents (players and the ball)
<li>each trajectory is a sequence of 121 timesteps which stores (x, y, v: speed)
<li>data = (N x 11 x 121 x 3) where N is the batch size
</ul>

<h2>Model Structure</h2>
<ol>
<li>Augmentation: disorder augmentation and random masking (80%) to produce two views of each possession (32x11x121x3) -> (64x11x121x3)
<li>Projection: project the 3 features to a larger dimension space of 128 (64x11x121x3) -> (64x11x121x128)
<li>Positional Encoding
<li>Temporal Transformer: explores the relationship between different timesteps within individual possessions
<li>Spatial Transformer: explores the relationship between different agents within individual possessions
<li>Decoders: calculates the loss of the model using three tasks
  <ol>
  <li>Motion Reconstruction: MLP 128->3(x, y, v)
  <li>Player Role Identification: MLP 128 -> 2(offense, defense)
  <li>Contrastive learning: MLP 128 -> 128 
  </ol>
</ol>
