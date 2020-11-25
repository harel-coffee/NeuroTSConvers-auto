import pandas as pd
from sklearn.cluster import KMeans

def save_files (data, filename):

	# Store data in csv files
	data. to_csv ("concat_time_series/%s.csv"%filename, sep = ';', index = False)
	data. to_pickle ("concat_time_series/%s.pkl"%filename)

def discr (x, seuil):
    if x < seuil:
        return 0
    else:
        return 1

if __name__ == '__main__':

    for file in ["behavioral_hh_data.pkl", "behavioral_hr_data.pkl"]:
        data = pd. read_pickle ("concat_time_series/" + file)

        for colname in data. columns:
            if "Overlap" in colname:
                data[colname] = data[colname]. apply (lambda x: discr (x, 0.085))

            #kmeans = KMeans(n_clusters=7, random_state=0).fit(data[colname]. values. reshape (-1,1))
            #data[colname] = kmeans. labels_

        save_files (data, file. split ('.')[0])
