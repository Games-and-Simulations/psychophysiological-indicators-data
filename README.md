# psychophysiological-indicators-data
Datasets from the experiments with psychophysiological indicators and their influence on EE user experience.
Dataset includes approx. 90 000 samples of physiological data measured using 3 different sensors and is gathered from 31 subjects. The measures are:

>1.id, 2.name (null values), 3.timedate, 4.enjoyment, 5.gender, 6.age, 7.video-games experience,  8.-11.blink frequency

For blinking frequency, except absolute value (reseting after each minute) there are derived features for delta of last 10, 30 and 60 seconds.
For the rest of the features, derived the measures represent their average, deviation, delta of last 10, 30 and 60 seconds:

>12.-14. attention, 15.-17.meditation, 18.-20.heart rate, 21.-23.heart rate amplitude, 24.-26.breathing rate, 27.-29.breathing rate amplitude, 30.-32.activity, 33.-35.gsr conductance, 36.-38.gsr resistance

BibTex cite:  
>@article{certicky2019psychophysiological,  
	title={Psychophysiological Indicators for Modeling User Experience in Interactive Digital Entertainment},  
	author={Certicky, Martin and Certicky, Michal and Magyar, G and Cavallo, F and Sincak, P and Vascak, J},  
	journal={sensors},  
	volume={19},  
	number={5},  
	year={2019},  
	publisher={Molecular Diversity Preservation International}  
}
