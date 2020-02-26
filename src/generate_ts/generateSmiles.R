
usage <- function (args)
{
	if (length (args) != 2)
	{
		stop ("arguments are not correct!, Usage: Rscript generateSimles PROJECT_PATH OUTPUT_DIR")
	}
}

args = commandArgs(trailingOnly=TRUE)
#print (as.list (args))
usage (args)

PROJECT_PATH = args[1]
OUTPUT_DIR = args[2]

print (PROJECT_PATH)
print (OUTPUT_DIR)

# Relative paths to global paths
CURRENT_PATH = getwd ()
PROJECT_PATH = paste0 (CURRENT_PATH, "/", PROJECT_PATH)
OUTPUT_DIR = paste0 (CURRENT_PATH, "/", OUTPUT_DIR)



temp = list.files (path = PROJECT_PATH, pattern = "*.csv", full.names = TRUE)

if (length (temp) == 0)
	stop ("Error, no csv file")

csv_filename = temp [1]
#corrected_csv_filename = tail (strsplit (csv_filename, '.')[[1]], n = 1)

#print (csv_filename)
corrected_csv_filename = strsplit (csv_filename, ".csv")[[1]]
corrected_csv_filename = paste0 (corrected_csv_filename, "_ofo.csv")

# We create temporarly a csv file with a name ending with _ofo_csv
system (paste ("cp", csv_filename, corrected_csv_filename))

#stop ()


## WARNING: You have to change the next line in order to match your own instal path
HMAD_DIRECTORY <- "/home/youssef/Documents/HMAD-master"; 
setwd(HMAD_DIRECTORY);

## Load the sources and packages
source("sourceR/loadSourcesAndPackages.R");

## Openface path
OPENFACE_DIRECTORY <- "/home/youssef/OpenFace";


## Define the software used to track the face
TRACKING_SOFTWARE <- "OPENFACE";

## List the HMAD projects
#lpn <- printProjectNames();

## Define the project and create it
## The project name, please use CAPITAL letters
#PROJECT_PATH <- "projects/TEST";
PROJECT_NAME = tail (strsplit (PROJECT_PATH, "/")[[1]], n = 1) #[length (strsplit (PROJECT_PATH, "/"))]

## Create the project
#lpl.R.dev.hmad.createNewProject(TRACKING_SOFTWARE, PROJECT_NAME);


## Create the OPENFACE output
#lpl.R.dev.hmad.createOpenFaceOutput(PROJECT_NAME, OPENFACE_DIRECTORY);


## Create Head Model and the internal facial landmarks residuals 
#df <- lpl.R.dev.hmad.createHeadModelAndResiduals(TRACKING_SOFTWARE, PROJECT_NAME);


filename <- paste(PROJECT_PATH, "/", PROJECT_NAME, "_ofo.csv", sep="");
csvprojectdir = PROJECT_NAME

# Openface csv file
csv <- read.table(filename, h=TRUE, sep=",");

tryCatch(
	expr = {

		df <- lpl.R.dev.openFaceOutputAnalysis.createHeadModelAndResiduals(PROJECT_PATH, csv);

		## Create the Action Units table 
		#audf <- lpl.R.dev.hmad.createActionUnitTable(TRACKING_SOFTWARE, PROJECT_NAME);
		audf <- lpl.R.dev.openFaceOutputAnalysis.createActionUnitTable(csv);

		## SMAD
		## Smile Movements Automatic Detection
		## Create the files for SMAD

		smad <- lpl.R.dev.smad.computeSMADTimeSeries (TRACKING_SOFTWARE, PROJECT_PATH, df, audf);
		write.table(smad, paste0(OUTPUT_DIR, "/", PROJECT_NAME, ".csv"), sep=";", row.names=FALSE);
		},
	error = function(e){
            		message('Caught an error!')
   		 	print(e)
			}
	)

system (paste0("rm -r ", PROJECT_PATH, "/model")) 
system (paste0("rm -r ", PROJECT_PATH, "/smile14")) 
system (paste0("rm -r ", PROJECT_PATH, "/tables")) 

# We delete the temporarly created file
system (paste0(PROJECT_PATH, "/*_ofo.csv"))
system (paste ("rm", corrected_csv_filename))

