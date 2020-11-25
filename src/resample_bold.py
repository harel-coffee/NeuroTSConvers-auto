import pandas as pd
import resampy

if __name__ == '__main__':
    physio_hh_data = pd.read_pickle ("concat_time_series/bold_hh_data.pkl")
    physio_hr_data = pd.read_pickle ("concat_time_series/bold_hr_data.pkl")

    behav_hh_data = pd.read_pickle ("concat_time_series/behavioral_hh_data.pkl")
    behav_hr_data = pd.read_pickle ("concat_time_series/behavioral_hr_data.pkl")

    colnames = physio_hh_data.columns

    ressampled_bold_hh = pd.DataFrame (resampy.resample (physio_hh_data. values, 2, 1, axis=0), columns = colnames)
    ressampled_bold_hr = pd.DataFrame (resampy.resample (physio_hr_data. values, 2, 1, axis=0), columns = colnames)

    ressampled_behav_hh = pd.DataFrame (resampy.resample (behav_hh_data. values, 2, 1, axis=0), columns = colnames)
    ressampled_behav_hr = pd.DataFrame (resampy.resample (behav_hr_data. values, 2, 1, axis=0), columns = colnames)

    #print (ressampled_hh. shape)
    #print (ressampled_hr. shape)
    ressampled_bold_hh. to_pickle ("resampled_time_series/bold_hh_data.pkl")
    ressampled_bold_hr. to_pickle ("resampled_time_series/bold_hr_data.pkl")

    ressampled_behav_hh. to_pickle ("resampled_time_series/ressampled_behav_hh.pkl")
    ressampled_behav_hr. to_pickle ("resampled_time_series/ressampled_behav_hr.pkl")
