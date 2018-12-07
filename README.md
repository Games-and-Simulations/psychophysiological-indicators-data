# psychophysiological-indicators-data
Datasets from the experiments with psychophysiological indicators and their influence on EE user experience.
Dataset includes approx. 90 000 samples of physiological data measured using 3 different sensors and is gathered from 31 subjects. The measures are:

-id
-name (null values)
-timedate
-gender
-age
-video-games experience
-blink frequency

For blinking frequency, except absolute value (reseting after each minute) there are derived features for delta of last 10, 30 and 60 seconds.
For the rest of the features, derived the measures represent their average, deviation, delta of last 10, 30 and 60 seconds:

-attention
-meditation
-heart rate
-heart rate amplitude
-breathing rate
-breathing rate amplitude
-activity
-gsr conductance
-gsr resistance
