# Specify the directory
dir <- "c:/paul/bojan/"

# Load the workspace
load(paste0(dir, "CommData.Rdata"))

# List all objects in the workspace
objects <- ls()

# Print the objects
print(objects)

# Assuming the dataset is named 'dataset', write it to a CSV file in the specified directory
#write.csv(dataset, file = paste0(dir, "CommData.csv"))