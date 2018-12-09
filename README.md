# psychophysiological-indicators-data
Datasets from the experiments with psychophysiological indicators and their influence on EE user experience.
Dataset includes approx. 90 000 samples of physiological data measured using 3 different sensors and is gathered from 31 subjects. The measures are:

>1.id, 2.name (null values), 3.timedate, 4.enjoyment, 5.gender, 6.age, 7.video-games experience,  8.-11.blink frequency

For blinking frequency, except absolute value (reseting after each minute) there are derived features for delta of last 10, 30 and 60 seconds.
For the rest of the features, derived the measures represent their average, deviation, delta of last 10, 30 and 60 seconds:

>12.-14. attention, 15.-17.meditation, 18.-20.heart rate, 21.-23.heart rate amplitude, 24.-26.breathing rate, 27.-29.breathing rate amplitude, 30.-32.activity, 33.-35.gsr conductance, 36.-38.gsr resistance

BibTex cite:  
>@inproceedings{certicky2018modeling,  
  title={Modeling user experience in electronic entertainment using psychophysiological measurements},  
  author={Certicky, Martin and Certicky, Michal and Sincak, Peter and Cavallo, Filippo},  
  booktitle={Proceedings of 2018 World Symposium on Digital Intelligence for Systems and Machines - DISA 2018},  
  pages={219--226},  
  year={2018},  
  publisher={Danvers}  
}
