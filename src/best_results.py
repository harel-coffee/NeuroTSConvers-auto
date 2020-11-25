import pandas as pd


#===================================================================
def res_all_models (type, files_dir):

    all_data = []
    evaluation_files = glob ("%s/*%s.tsv*"%(files_dir, type))

    for file in evaluation_files:
        model_name = get_model_name (file)
        data = pd.read_csv (file, sep = '\t', header = 0, na_filter = False, index_col=False)

        if data. shape [0] == 0:
            continue

        data. sort_values (["region"], inplace = True)

        data = data. assign (model = lambda x : model_name)
        if len (all_data) == 0:
            all_data = data
        else:
            all_data = pd. concat([all_data, data], axis = 0)

    all_data. reset_index (inplace = True)
    #all_data = all_data. loc [all_data. groupby ("region") ["fscore. mean"]. idxmax (), :]

    return all_data

if __name__ == '__main__':

	parser = argparse. ArgumentParser ()
	parser. add_argument ('--regions','-rg', nargs = '+', type=int, default = [1, 2, 3, 4, 5, 6])
	parser. add_argument ('--model','-ml', help = "prediction model", default = "RF")
	parser. add_argument ("--crossv", "-cv", help = "handle cross-validation results", action="store_true")
	parser. add_argument ("--baseline", "-b", help = "compare the best model with the baseline", action="store_true")
	args = parser.parse_args()

	brain_areas_desc = pd. read_csv ("brain_areas.tsv", sep = '\t', header = 0)
	brain_areas = []

	print (args. model)

	for num_region in args. regions:
		brain_areas. append (brain_areas_desc . loc [brain_areas_desc ["Code"] == num_region, "Name"]. values [0])

	#get_best_predictions (type, files_dir)

	if args.crossv:
	    _files_dir = "results/models_params"
	else:
	    _files_dir = "results/prediction"


	df_hh = evaluate_baseline (brain_areas, type = "HH", files_dir = _files_dir)
	df_hr = evaluate_baseline (brain_areas, type = "HR", files_dir = _files_dir)
