# cnn-dssm
A tensorflow version of dssm using textcnn as feature extractor.
## notations
### Q means query words, here using document's title
### D+ means hit docs, here using document's content
### D- mean negative docs, here using negative document's content
### p(Q,D+) = cosine(Q,D+)

## cnn_dssm.py is version 1
  loss function =  - p(Q,D+)
## cdssm2.py is version 2
  loss function = max{1 - p(Q,D+) + p(Q,D-),0}

## tricks
In my practice:
    little learning_rate, such as 0.001, easier to converge
    small conv output,such as 64, easier to training
    active function tanh get better,relu usally got nan loss
